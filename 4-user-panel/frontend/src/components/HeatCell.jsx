export default function HeatCell({possibility}) {
  return(
    <div className={`bg-heat-${Math.floor(possibility * 10) + 1} h-3 w-3 rounded-sm`}></div>
  )
}