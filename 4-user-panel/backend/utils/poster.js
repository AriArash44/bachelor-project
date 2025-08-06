import axios from "axios";

const postData = async (url, payload) => {
  try {
    const response = await axios.post(url, payload);
    return response.data;
  } catch (error) {
    console.error('Axios error:', error);
    throw error;
  }
};

export default postData;