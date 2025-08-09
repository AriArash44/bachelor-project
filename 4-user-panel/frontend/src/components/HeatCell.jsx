import Tooltip from "./Tooltip.jsx";

export default function HeatCell({ possibility }) {
  const level = Math.min(10, Math.max(1, Math.floor(possibility * 10) + 1));
  return (
    <Tooltip text={`${(possibility * 100).toFixed(2)}%`} position="top">
      <div className={`bg-heat-${level} h-3 w-3 rounded-sm`} />
    </Tooltip>
  );
}
