import { useEffect, useRef, useState, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

// Fades/slides a block in the first time it scrolls into view, then leaves
// it alone — no re-hiding on scroll-away, which would be distracting rather
// than polished. Degrades to "just show it" when IntersectionObserver isn't
// available (older browsers, and the happy-dom test environment), so this
// is a progressive enhancement, not a hard dependency for content to appear.
export function Reveal({ children }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") {
      setVisible(true);
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.12, rootMargin: "0px 0px -64px 0px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={ref} className={`reveal${visible ? " is-visible" : ""}`}>
      {children}
    </div>
  );
}
