import Table from "./components/Table";
import { useFetcher } from "./hooks/useFetcher";
import { useState, useEffect } from "react";
import { formatTimeStamp } from "./utils/formatDate";

function App() {
  const { data } = useFetcher('/get-predictions')
  const [ showData, setShowData ] = useState({});
  useEffect(() => {
    if(data) {
      const finalData = [];
      data.result.map((datium) => {
        finalData.push(
          {
            "request_id" : datium.activity_id,
            "timestamp" : formatTimeStamp(datium.activity_timestamp),
            "final_label" : datium.predicted,
            "confidence_score" : datium.confidence,
            "datails": "/images/magnify.png"
          }
        )
      })
    }
    else{
      setShowData({});
    }
  }, [data])
  return (
    <div className="bg-white rounded-3xl p-6 min-h-[50vh] max-h-[80vh] w-1/2 flex flex-col">
      <h1 className="text-neutral-300 font-medium flex-1/12">Activities Results</h1>
      {showData ? 
        <></> 
      : 
        <div className="flex justify-center items-center">
          <p className="text-4xl text-neutral-200">No Activity to show yet!</p>
        </div>
      }
    </div>
  )
}

export default App