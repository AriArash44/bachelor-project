import Table from "./components/Table";
import { useFetcher } from "./hooks/useFetcher";
import { useState, useEffect, useMemo } from "react";
import { formatTimeStamp } from "./utils/formatDate";

function App() {
  const { data } = useFetcher('get-predictions')
  const [ showData, setShowData ] = useState([]);
  useEffect(() => {
    if(data) {
      const finalData = [];
      data.results.map((datium) => {
        finalData.push(
          {
            "request_id" : datium.activity_id,
            "timestamp" : formatTimeStamp(datium.activity_timestamp),
            "final_label" : datium.predicted,
            "confidence_score" : datium.confidence,
            "details": true
          }
        )
      })
      setShowData(finalData);
    }
    else{
      setShowData({});
    }
  }, [data])
  const headers = useMemo(
    () => (showData.length > 0 ? Object.keys(showData[0]) : []),
    [showData]
  );
  return (
    <div className="bg-white rounded-3xl p-6 min-h-[50vh] max-h-[80vh] w-1/2 flex flex-col overflow-scroll">
      <h1 className="text-neutral-300 font-medium flex-1/12 px-6">Activities Result</h1>
      {showData.length > 0 ? (
        <Table headers={headers} data={showData} columnWeights={[1, 2, 1, 1.5, 1]}/>
      ) : (
        <div className="flex flex-[100] items-center justify-center">
          <p className="text-4xl text-neutral-200 font-light">No Activity to show yet :(</p>
        </div>
      )}
    </div>
  )
}

export default App