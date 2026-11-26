// Semicircular probability gauge with the calibrated decision-threshold marker.
export default function Gauge({ value, threshold = 0.5, positive }) {
  const r = 70, cx = 90, cy = 86;
  const angle = (v) => Math.PI * (1 - v);
  const pt = (v, rr = r) => [cx + rr * Math.cos(angle(v)), cy - rr * Math.sin(angle(v))];
  const arc = (a, b) => {
    const [x1, y1] = pt(a), [x2, y2] = pt(b);
    return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`;
  };
  const [tx1, ty1] = pt(threshold, r - 14), [tx2, ty2] = pt(threshold, r + 10);
  const color = value >= threshold ? "#4f46e5" : "#94A3B8";
  return (
    <svg viewBox="0 -14 180 120" className="w-full max-w-[220px]" role="img"
      aria-label={`probability ${(value * 100).toFixed(1)} percent, threshold ${(threshold * 100).toFixed(0)} percent`}>
      <path d={arc(0, 1)} stroke="#E2E8F0" strokeWidth="14" fill="none" strokeLinecap="round" />
      {value > 0.005 && <path d={arc(0, Math.min(value, 0.999))} stroke={color} strokeWidth="14" fill="none" strokeLinecap="round" />}
      <line x1={tx1} y1={ty1} x2={tx2} y2={ty2} stroke="#1B1F3B" strokeWidth="2.5" />
      <text x={cx} y={cy - 12} textAnchor="middle" className="font-display" fontSize="26" fontWeight="700" fill="#1B1F3B">
        {(value * 100).toFixed(0)}%
      </text>
      <text x={cx} y={cy + 6} textAnchor="middle" fontSize="9.5" fill="#64748B">P({positive})</text>
      <text x={tx2} y={ty2 - 3} textAnchor="middle" fontSize="8.5" fill="#1B1F3B">t={threshold.toFixed(2)}</text>
    </svg>
  );
}
