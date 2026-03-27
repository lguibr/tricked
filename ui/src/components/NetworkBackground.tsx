import { useEffect, useRef } from 'react';

export function NetworkBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let particles: { x: number; y: number; z: number; vx: number; vy: number; vz: number }[] = [];
    let animationFrameId: number;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    const init = () => {
      resize();
      particles = [];
      const particleCount = Math.floor((window.innerWidth * window.innerHeight) / 12000);
      for (let i = 0; i < particleCount; i++) {
        particles.push({
          x: (Math.random() - 0.5) * canvas.width * 2,
          y: (Math.random() - 0.5) * canvas.height * 2,
          z: Math.random() * 1000,
          vx: (Math.random() - 0.5) * 1.5,
          vy: (Math.random() - 0.5) * 1.5,
          vz: (Math.random() - 0.5) * 2.0,
        });
      }
    };

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#050505';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const connectDistanceSq = 200 * 200;
      const projected: { x: number; y: number; z: number; scale: number }[] = [];

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy;
        p.z += p.vz;

        if (p.x < -canvas.width) p.vx *= -1;
        if (p.x > canvas.width) p.vx *= -1;
        if (p.y < -canvas.height) p.vy *= -1;
        if (p.y > canvas.height) p.vy *= -1;
        if (p.z < 10) p.vz *= -1;
        if (p.z > 1500) p.vz *= -1;

        const fov = 400;
        const scale = fov / (fov + p.z);
        const x2d = canvas.width / 2 + p.x * scale;
        const y2d = canvas.height / 2 + p.y * scale;

        projected.push({ x: x2d, y: y2d, z: p.z, scale });
      }

      for (let i = 0; i < projected.length; i++) {
        const p1 = projected[i];
        if (p1.x < 0 || p1.x > canvas.width || p1.y < 0 || p1.y > canvas.height) continue;

        ctx.beginPath();
        ctx.arc(p1.x, p1.y, 1.5 * p1.scale, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(16, 185, 129, ${p1.scale * 0.8})`; // neon green points
        ctx.fill();

        const neighbors = [];
        for (let j = i + 1; j < projected.length; j++) {
          const p2 = projected[j];
          const dx = p1.x - p2.x;
          const dy = p1.y - p2.y;
          const distSq = dx * dx + dy * dy;

          if (distSq < connectDistanceSq * p1.scale) {
            neighbors.push(p2);
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = `rgba(16, 185, 129, ${p1.scale * 0.15})`;
            ctx.stroke();
          }
        }

        for (let a = 0; a < neighbors.length; a++) {
          for (let b = a + 1; b < neighbors.length; b++) {
            const na = neighbors[a];
            const nb = neighbors[b];
            const dx = na.x - nb.x;
            const dy = na.y - nb.y;
            if ((dx * dx + dy * dy) < connectDistanceSq * p1.scale) {
              ctx.beginPath();
              ctx.moveTo(p1.x, p1.y);
              ctx.lineTo(na.x, na.y);
              ctx.lineTo(nb.x, nb.y);
              ctx.closePath();
              ctx.fillStyle = `rgba(234, 179, 8, ${p1.scale * 0.06})`; // neon yellow triangles
              ctx.fill();
            }
          }
        }
      }
      animationFrameId = requestAnimationFrame(draw);
    };

    window.addEventListener('resize', init);
    init();
    draw();

    return () => {
      window.removeEventListener('resize', init);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return <canvas ref={canvasRef} className="fixed inset-0 pointer-events-none z-0" />;
}
