import { Router } from 'express';
import db from '../utils/dbPool.js';

const router = Router();

router.get('/get-predictions', async (req, res) => {
  try {
    const conn = await db.getConnection();
    const [predictions] = await conn.execute(`SELECT * FROM Activities`);
    await Promise.all(predictions.map(async (pred) => {
      const [probaRows] = await conn.execute(
        `SELECT vector_id FROM Vectors 
         WHERE activity_id = ? AND vector_type = 'proba'`,
        [pred.activity_id]
      );
      if (!probaRows.length) return;
      const [confidenceRows] = await conn.execute(
        `SELECT probability FROM VectorFeatures
        WHERE vector_id = ? AND label = ?`,
        [probaRows[0].vector_id, pred.predicted]
      );
      if (confidenceRows.length) {
        pred.confidence = (confidenceRows[0].probability * 100).toFixed(2) + "%";
      }
    }));
    conn.release();
    res.json({ results: predictions });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'A server error occurred!' });
  }
});

router.get('/get-possibilities', async (req, res) => {
  try {
    const { id } = req.query;
    if (!id) throw new Error("id is required");
    const probabilities = {};
    const conn = await db.getConnection();
    const [vectors] = await conn.execute(
      `SELECT vector_id, vector_type FROM Vectors WHERE activity_id = ?`,
      [id]
    );
    await Promise.all(vectors.map(async (vector) => {
      const [vectorValues] = await conn.execute(
        `SELECT label, probability FROM VectorFeatures WHERE vector_id = ?`,
        [vector.vector_id]
      );
      const vectorOutput = {};
      vectorValues.forEach(({ label, probability }) => {
        vectorOutput[label] = parseFloat((probability).toFixed(4));
      });
      probabilities[vector.vector_type] = vectorOutput;
    }));
    conn.release();
    res.json(probabilities);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'A server error occurred!' });
  }
});


export default router;