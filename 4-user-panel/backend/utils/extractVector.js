const extractVectorByPrefix = (obj, prefix) => {
  const result = {};
  for (const [key, value] of Object.entries(obj)) {
    if (key.startsWith(prefix + '_')) {
      result[key.replace(prefix + '_', '')] = value;
    }
  }
  return result;
}

export default extractVectorByPrefix;