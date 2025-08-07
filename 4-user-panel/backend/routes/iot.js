import { Router } from 'express';
import multer from 'multer';
import { Readable } from 'stream';
import FormData from 'form-data';
import csvParser from 'csv-parser';
import postData from '../utils/poster.js';
import db from '../utils/dbPool.js';
import extractVectorByPrefix from '../utils/extractVector.js';

const datetimeFields = [
  'fridge.datetime',
  'garage_door.datetime',
  'gps_tracker.datetime',
  'modbus.datetime',
  'motion_light.datetime',
  'thermostat.datetime',
  'weather.datetime'
];

const upload = multer();  
const router = Router();

router.post('/predict', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'input file is required.' });
  }

  const rows = [];
  await new Promise((resolve, reject) => {
    Readable.from(req.file.buffer)
      .pipe(csvParser())
      .on('data', row => rows.push(row))
      .on('end', resolve)
      .on('error', reject);
  });

  const form = new FormData();
  form.append('file', req.file.buffer, {
    filename: req.file.originalname,
    contentType: req.file.mimetype
  });

  let predictions;
  try {
    const result = await postData(
      'http://ai:5000/predict',
      form,
      { headers: form.getHeaders() }
    );
    predictions = result["prediction"];
  } catch (err) {
    console.error('Error calling Python service:', err);
    return res.status(502).json({ error: 'Prediction service error.' });
  }

  const conn = await db.getConnection();
  await conn.beginTransaction();
  try {
    const summary = [];
    for (let i = 0; i < rows.length; i++) {
      const csvRow = rows[i];
      const prediction = predictions[i];
      const dev = extractVectorByPrefix(prediction, 'dev');
      const lin = extractVectorByPrefix(prediction, 'lin');
      const net = extractVectorByPrefix(prediction, 'net');
      const proba = extractVectorByPrefix(prediction, 'proba');

      const predicted = prediction.predicted;
      const confidence = proba[predicted];

      let rawTimestamp = null;
      for (const field of datetimeFields) {
        if (csvRow[field] && csvRow[field].trim()) {
          rawTimestamp = csvRow[field].trim();
          break;
        }
      }
      if (!rawTimestamp) {
        throw new Error(`No valid timestamp found for row index ${i}`);
      }
      const timestamp = new Date(rawTimestamp);

      const [actResult] = await conn.execute(
        `INSERT INTO Activities (activity_timestamp, predicted)
         VALUES (?, ?)`,
        [ new Date(timestamp), predicted ]
      );
      const activityId = actResult.insertId;

      for (const [vectorType, vectorData] of Object.entries({ dev, lin, net, proba })) {
        const [vecResult] = await conn.execute(
          `INSERT INTO Vectors (activity_id, vector_type)
           VALUES (?, ?)`,
          [ activityId, vectorType ]
        );
        const vectorId = vecResult.insertId;
        for (const [label, prob] of Object.entries(vectorData)) {
          await conn.execute(
            `INSERT INTO VectorFeatures (vector_id, label, probability)
             VALUES (?, ?, ?)`,
            [ vectorId, label, prob ]
          );
        }
      }

      summary.push({ predicted, confidence });
    }
    await conn.commit();
    res.json({ results: summary });
  } catch (dbErr) {
    await conn.rollback();
    console.error('DB error:', dbErr);
    res.status(500).json({ error: 'Database transaction failed.' });
  } finally {
    conn.release();
  }
});

export default router;
