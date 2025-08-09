import { useState } from 'react';

const Tooltip = ({ children, text, position = 'top' }) => {
  const [visible, setVisible] = useState(false);

  const positionClasses = {
    top: 'bottom-full left-1/2 transform -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 transform -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 transform -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 transform -translate-y-1/2 ml-2',
  };

  const arrowStyles = {
    top: {
      left: '50%',
      bottom: '-5px',
      transform: 'translateX(-50%) rotate(45deg)',
    },
    bottom: {
      left: '50%',
      top: '-5px',
      transform: 'translateX(-50%) rotate(45deg)',
    },
    left: {
      top: '50%',
      right: '-5px',
      transform: 'translateY(-50%) rotate(45deg)',
    },
    right: {
      top: '50%',
      left: '-5px',
      transform: 'translateY(-50%) rotate(45deg)',
    },
  };

  return (
    <div
      className="relative inline-block"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      {children}
      {visible && (
        <div
          className={`absolute z-10 bg-gray-800 text-white font-light text-sm px-3 py-2 rounded-md shadow-lg whitespace-nowrap opacity-90 ${positionClasses[position]}`}
        >
          {text}
          <div
            style={{
              position: 'absolute',
              width: '10px',
              height: '10px',
              backgroundColor: '#1f2937',
              ...arrowStyles[position],
            }}
            className="z-0"
          />
        </div>
      )}
    </div>
  );
};

export default Tooltip;
