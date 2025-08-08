const Table = ({ headers, data }) => {
  return (
    <div className="space-y-2">
      <div className="flex justify-evenly items-center font-semibold text-neutral-200">
        {headers.map((header) => (
          <p key={header}>{header}</p>
        ))}
      </div>
      {data.map((datium, index) => (
        <div
          key={index}
          className="flex justify-evenly items-center border-t border-neutral-100 pt-2"
        >
          {headers.map((header) => {
            { header === "details" ? 
              <p key={header} className="text-neutral-100">
                {datium[header]}
              </p> 
            : 
              <button className="cursor-pointer" bg-image={datium.details}></button>
            }
          })}
        </div>
      ))}
    </div>
  );
};

export default Table;
