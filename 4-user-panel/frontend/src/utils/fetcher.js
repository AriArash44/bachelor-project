import axios from 'axios';

export const axiosFetcher = async (endpoint) => {
  try {
    const url = import.meta.env.VITE_BASE_URL + endpoint;
    const response = await axios.get(url);
    return response.data;
  } catch (error) {
    console.error(error.response?.data || error.message);
    throw error;
  }
}