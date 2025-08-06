import { Router } from 'express';
import db from '../utils/dbPool.js';

const router = Router();

router.get('/get-predictions', (req, res) => {

})

router.post('/get-possibilities', (req, res) => {
  const { id } = req.query;
})

export default router;