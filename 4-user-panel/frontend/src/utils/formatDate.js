export const formatTimeStamp = (str) => {
  const [date, time] = str.split("T");
  return date + " " + time.split(".")[0];
}