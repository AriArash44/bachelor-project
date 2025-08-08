import { useFetch } from "../hooks/useFetch.js";
import React, { useState, useEffect, useContext } from "react";
import { ModalContext } from '../contexts/modalContext.js';
import Modal from "./Modal.jsx";
import HeatCell from "../components/HeatCell.jsx"

const Table = ({ headers, data, columnWeights }) => {
  const [detailsEndpoint, setDetailsEndpoint] = useState(null);
  const { data: detailData, loading, _error } = useFetch(detailsEndpoint);
  const { isOpen, openModal } = useContext(ModalContext);
  useEffect(() => {
    if (!loading && detailData) {
      openModal(true);
    }
  }, [loading, detailData]);
  const gridTemplate = columnWeights
    ? columnWeights.map((w) => `${w}fr`).join(' ')
    : `repeat(${headers.length}, 1fr)`;
  return (
    <div className="space-y-2 mt-10 w-full overflow-scroll">
      <div
        className="grid text-neutral-200 text-left"
        style={{ gridTemplateColumns: gridTemplate }}
      >
        {headers.map((header) => (
          <p key={header} className="px-2 py-1 text-center">{header}</p>
        ))}
      </div>
      <div>
        {data.map((datium, index) => (
          <div
            key={index}
            className="grid items-center border-t border-neutral-100 pt-2"
            style={{ gridTemplateColumns: gridTemplate }}
          >
            {headers.map((header) =>
              header === "details" ? (
                <button
                  key={header}
                  className="cursor-pointer bg-contain bg-no-repeat
                  flex justify-center items-center"
                  onClick={() => {
                    setDetailsEndpoint(datium[header]);
                  }}
                >
                  <img src="/images/magnify.png" alt="details" className="h-4 w-4" />
                </button>
              ) : (
                <p key={header} className="text-neutral-200 px-2 py-1 text-center">
                  {datium[header]}
                </p>
              )
            )}
          </div>
        ))}
      </div>
      {isOpen && (
        <Modal>
          <div className="w-full">
            <div
              className="grid"
              style={{
                gridTemplateColumns: "repeat(11, 20px)",
                gridTemplateRows: "repeat(11, 20px)",
              }}
            >
              <div></div>
              {Object.keys(detailData.dev).map((attackType) => (
                <div
                  key={attackType + "_label"}
                  className="text-neutral-300 text-xs font-medium -rotate-45 origin-bottom-left flex items-center justify-center"
                >
                  {attackType}
                </div>
              ))}
              {["dev", "lin", "net", "proba"].map((deviceKey) => (
                <React.Fragment key={deviceKey}>
                  <div className="text-center text-neutral-300 font-medium flex items-center justify-center">
                    {{
                      dev: "devices",
                      lin: "linux",
                      net: "network",
                      proba: "aggregated",
                    }[deviceKey]}
                  </div>
                  {Object.keys(detailData.dev).map((attackType) => (
                    <div key={`${deviceKey}_${attackType}`} className="flex justify-center items-center">
                      <HeatCell possibility={detailData[deviceKey][attackType]} />
                    </div>
                  ))}
                </React.Fragment>
              ))}
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};

export default Table;