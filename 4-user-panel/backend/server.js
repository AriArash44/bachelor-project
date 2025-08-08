import express from 'express';
import cors from 'cors';
import iotRouter from './routes/iot.js';
import uiRouter from './routes/ui.js';

const app = express();
const PORT = 8000;

app.use(cors({
  origin: 'http://localhost:3000',
  credentials: true
}));
app.use(express.json());

app.use('/api', iotRouter);
app.use('/api', uiRouter);

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));