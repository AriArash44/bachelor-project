const Table = ({ headers, data, columnWeights }) => {
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
              header === "details" && datium[header] ? (
                <button
                  key={header}
                  className="cursor-pointer bg-contain bg-no-repeat
                  flex justify-center items-center"
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
    </div>
  );
};


export default Table;