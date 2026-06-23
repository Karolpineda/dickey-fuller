import React, { useState, useEffect, useMemo, useRef } from 'react';

// Math Normal noise generator (Box-Muller Transform)
function randomNormal() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// AR(1) Simulator: y_t = alpha + beta*t + rho*y_{t-1} + e_t
function generateAR1(T, rho, drift, trend, noiseStd) {
  const series = [];
  let y_prev = 0;
  for (let t = 1; t <= T; t++) {
    const error = randomNormal() * noiseStd;
    const y_curr = drift + trend * (t / T) + rho * y_prev + error;
    series.push(y_curr);
    y_prev = y_curr;
  }
  return series;
}

// OLS regression: Y_i = beta0 + beta1 * X_i
function calculateDF(series) {
  const m = series.length;
  if (m < 3) return null;

  // X_i = Y_{t-1}, Y_i = \Delta Y_t = Y_t - Y_{t-1}
  const X = [];
  const Y = [];
  for (let i = 1; i < m; i++) {
    X.push(series[i - 1]);
    Y.push(series[i] - series[i - 1]);
  }
  const n = X.length;

  let sum_X = 0;
  let sum_Y = 0;
  for (let i = 0; i < n; i++) {
    sum_X += X[i];
    sum_Y += Y[i];
  }
  const mean_X = sum_X / n;
  const mean_Y = sum_Y / n;

  const X_diff = [];
  const Y_diff = [];
  const prod = [];
  const sq = [];
  let sum_prod = 0;
  let sum_sq = 0;

  for (let i = 0; i < n; i++) {
    const xd = X[i] - mean_X;
    const yd = Y[i] - mean_Y;
    X_diff.push(xd);
    Y_diff.push(yd);
    prod.push(xd * yd);
    sq.push(xd * xd);
    sum_prod += xd * yd;
    sum_sq += xd * xd;
  }

  const beta1 = sum_sq !== 0 ? sum_prod / sum_sq : 0;
  const beta0 = mean_Y - beta1 * mean_X;

  const Y_pred = [];
  const resids = [];
  const resids_sq = [];
  let sum_resids_sq = 0;

  for (let i = 0; i < n; i++) {
    const pred = beta0 + beta1 * X[i];
    const resid = Y[i] - pred;
    Y_pred.push(pred);
    resids.push(resid);
    resids_sq.push(resid * resid);
    sum_resids_sq += resid * resid;
  }

  const df = n - 2; // n observations - 2 estimated parameters (intercept & slope)
  const sigma_sq = df > 0 ? sum_resids_sq / df : 0;
  const var_beta1 = sum_sq !== 0 ? sigma_sq / sum_sq : 0;
  const se_beta1 = Math.sqrt(var_beta1);
  const t_stat = se_beta1 !== 0 ? beta1 / se_beta1 : 0;

  return {
    X, Y, n,
    sum_X, mean_X,
    sum_Y, mean_Y,
    X_diff, Y_diff,
    prod, sq,
    sum_prod, sum_sq,
    beta1, beta0,
    Y_pred, resids, resids_sq,
    sum_resids_sq,
    df, sigma_sq,
    var_beta1, se_beta1,
    t_stat
  };
}

// Icons
const BookIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
  </svg>
);

const CalcIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="4" y="2" width="16" height="20" rx="2" ry="2" />
    <line x1="8" y1="6" x2="16" y2="6" />
    <line x1="16" y1="14" x2="16" y2="18" />
    <line x1="8" y1="10" x2="8" y2="10" />
    <line x1="12" y1="10" x2="12" y2="10" />
    <line x1="16" y1="10" x2="16" y2="10" />
    <line x1="8" y1="14" x2="8" y2="14" />
    <line x1="12" y1="14" x2="12" y2="14" />
    <line x1="8" y1="18" x2="8" y2="18" />
    <line x1="12" y1="18" x2="12" y2="18" />
  </svg>
);

const SimIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="4" y1="21" x2="4" y2="14" />
    <line x1="4" y1="10" x2="4" y2="3" />
    <line x1="12" y1="21" x2="12" y2="12" />
    <line x1="12" y1="8" x2="12" y2="3" />
    <line x1="20" y1="21" x2="20" y2="16" />
    <line x1="20" y1="12" x2="20" y2="3" />
    <line x1="2" y1="14" x2="6" y2="14" />
    <line x1="10" y1="8" x2="14" y2="8" />
    <line x1="18" y1="16" x2="22" y2="16" />
  </svg>
);

const CaseIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
    <polyline points="10 9 9 9 8 9" />
  </svg>
);

const ShieldIcon = ({ isGreen }) => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    {isGreen ? (
      <path d="M9 11l2 2 4-4" />
    ) : (
      <>
        <line x1="12" y1="9" x2="12" y2="13" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </>
    )}
  </svg>
);

function App() {
  const [activeTab, setActiveTab] = useState('theory');

  // MathJax typesetting effect
  useEffect(() => {
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise().catch((err) =>
        console.error('MathJax error during render: ', err)
      );
    }
  });

  return (
    <>
      <header className="app-header">
        <div className="brand">
          <span className="brand-accent">Dickey-Fuller</span>
        </div>
        <nav className="nav-tabs">
          <button
            className={`tab-btn ${activeTab === 'theory' ? 'active' : ''}`}
            onClick={() => setActiveTab('theory')}
          >
            <BookIcon /> Fundamentos Teóricos
          </button>
          <button
            className={`tab-btn ${activeTab === 'calculator' ? 'active' : ''}`}
            onClick={() => setActiveTab('calculator')}
          >
            <CalcIcon /> Calculadora a Mano
          </button>
          <button
            className={`tab-btn ${activeTab === 'simulation' ? 'active' : ''}`}
            onClick={() => setActiveTab('simulation')}
          >
            <SimIcon /> Simulador AR(1)
          </button>
          <button
            className={`tab-btn ${activeTab === 'case_study' ? 'active' : ''}`}
            onClick={() => setActiveTab('case_study')}
          >
            <CaseIcon /> Caso Práctico
          </button>
        </nav>
      </header>

      <div className="hero">
        <h1>La Prueba de Dickey-Fuller</h1>
        <p>
          Fundamentos matemáticos y análisis interactivo paso a paso para la detección de raíces unitarias y estacionariedad.
        </p>
        <div className="hero-group">
          <span className="hero-group-title">Grupo 10</span>
          <div className="hero-group-members">
            <span>Chicaiza Eduardo</span>
            <span>Navarrete Marlon</span>
            <span>Pineda Fabricio</span>
            <span>Soria Samanta</span>
            <span>Tapia Alex</span>
          </div>
        </div>
        <div className="hero-chips">
          <span className="hero-chip special">Interacciones Reactivas</span>
          <span className="hero-chip">Nobel 2003 (Granger &amp; Engle)</span>
          <span className="hero-chip">MacKinnon Critical Values</span>
          <span className="hero-chip">Simulaciones Monte Carlo</span>
        </div>
      </div>

      <main className="main-content">
        {activeTab === 'theory' && <TheoryTab />}
        {activeTab === 'calculator' && <CalculatorTab />}
        {activeTab === 'simulation' && <SimulationTab />}
        {activeTab === 'case_study' && <CaseStudyTab />}
      </main>

      <footer className="app-footer">
        <p>
          <strong>Dickey &amp; Fuller</strong>
        </p>
      </footer>
    </>
  );
}

/* ============================================================================
   THEORY TAB
   ============================================================================ */
function TheoryTab() {
  const [activeSec, setActiveSec] = useState('proposito');
  const containerRef = useRef(null);

  // MathJax typesetting effect specifically for changing slides
  useEffect(() => {
    if (window.MathJax && window.MathJax.typesetPromise && containerRef.current) {
      window.MathJax.typesetClear([containerRef.current]);
      window.MathJax.typesetPromise([containerRef.current]).catch((err) =>
        console.error('MathJax error during render: ', err)
      );
    }
  }, [activeSec]);

  const sections = [
    { id: 'proposito', label: '01 · Propósito de la Prueba' },
    { id: 'problema', label: '02 · El Problema' },
    { id: 'estacionariedad', label: '03 · Proceso AR(1)' },
    { id: 'formulacion', label: '04 · Formulación Original' },
    { id: 'modelos', label: '05 · Los Tres Modelos' },
    { id: 'tendencia', label: '06 · Modelo con Tendencia' },
    { id: 'ampliada', label: '07 · Prueba Ampliada' },
    { id: 'estadistico', label: '08 · Estadístico de Prueba' },
    { id: 'criterio', label: '09 · Criterio de Decisión' }
  ];

  const activeIndex = sections.findIndex((sec) => sec.id === activeSec);

  return (
    <div className="theory-layout tab-panel">
      <aside className="theory-toc">
        <div className="theory-toc-title">Diapositivas</div>
        <ul className="toc-list">
          {sections.map((sec) => (
            <li key={sec.id}>
              <a
                className={`toc-link ${activeSec === sec.id ? 'active' : ''}`}
                onClick={() => setActiveSec(sec.id)}
              >
                {sec.label}
              </a>
            </li>
          ))}
        </ul>
      </aside>

      <div className="theory-sections" ref={containerRef}>
        <div className="step-card" key={activeSec}>
          {activeSec === 'proposito' && (
            <section className="theory-section" id="proposito">
              <span className="section-tag">Diapositiva 1 de 9 — Propósito de la prueba</span>
              <h2>Propósito de la prueba</h2>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginTop: '1.25rem' }}>
                <p style={{ fontSize: '0.95rem', lineHeight: '1.6', color: 'var(--text-light)', margin: 0 }}>
                  Llamada de esta forma por los estadísticos estadounidenses <strong>David Dickey</strong> y <strong>Wayne Fuller</strong>, esta prueba de raíz única detecta estadísticamente la presencia de conducta tendencial que estocástica en las series temporales de las variables a través de contrastar las hipótesis.
                </p>

                <div className="grid-2" style={{ gridTemplateColumns: '1fr 1fr', gap: '1.25rem' }}>
                  <div className="glass-card" style={{ borderLeft: '3px solid var(--accent)', padding: '1.25rem', margin: '0' }}>
                    <h4 style={{ color: 'var(--accent)', marginBottom: '0.5rem', textTransform: 'uppercase', fontSize: '0.8rem', letterSpacing: '0.05em' }}>
                      Objetivo
                    </h4>
                    <p style={{ fontSize: '0.85rem', lineHeight: '1.5', margin: 0 }}>
                      Este contraste permite saber o conocer si existe presencia significativa de tendencia en las series temporales de las variables.
                    </p>
                  </div>

                  <div className="glass-card" style={{ borderLeft: '3px solid var(--accent2)', padding: '1.25rem', margin: '0' }}>
                    <h4 style={{ color: 'var(--accent2)', marginBottom: '0.5rem', textTransform: 'uppercase', fontSize: '0.8rem', letterSpacing: '0.05em' }}>
                      Determinación Clave
                    </h4>
                    <p style={{ fontSize: '0.85rem', lineHeight: '1.5', margin: 0 }}>
                      En otras palabras, la prueba se utiliza para determinar si una raíz unitaria se encuentra presente en un modelo autorregresivo.
                    </p>
                  </div>
                </div>

                <div className="glass-card" style={{ borderTop: '3px solid var(--green)', padding: '1.25rem', margin: '0' }}>
                  <h4 style={{ color: 'var(--green)', marginBottom: '0.5rem', textTransform: 'uppercase', fontSize: '0.8rem', letterSpacing: '0.05em' }}>
                    Ampliación y Evolución
                  </h4>
                  <p style={{ fontSize: '0.85rem', lineHeight: '1.5', margin: 0 }}>
                    Es importante destacar que, los mismos estadísticos ampliaron su prueba básica de raíz unitaria autorregresiva, para que pudiera adaptarse a modelos con mayor complejidad <strong>(prueba Dickey-Fuller aumentada)</strong>, que es la que prueba una raíz unitaria en una muestra de serie temporal.
                  </p>
                </div>
              </div>
            </section>
          )}

          {activeSec === 'problema' && (
            <section className="theory-section" id="problema">
              <span className="section-tag">Diapositiva 2 de 9 — El Problema</span>
              <h2>¿Por qué existe la Prueba de Dickey-Fuller?</h2>

              <div className="grid-2" style={{ gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginTop: '1rem' }}>
                <div className="glass-card" style={{ borderTop: '3px solid var(--danger)', padding: '1.25rem', margin: '0' }}>
                  <h4 style={{ color: 'var(--danger)', marginBottom: '0.5rem', textTransform: 'uppercase', fontSize: '0.8rem', letterSpacing: '0.05em' }}>
                    EL PROBLEMA: Regresión Espuria
                  </h4>
                  <p style={{ fontSize: '0.85rem', marginBottom: '0.75rem' }}>
                    <strong>Granger &amp; Newbold (1974)</strong> {"demostraron que regresionar dos caminatas aleatorias completamente independientes produce un $R^2 \\approx 1$ y t-estadísticos muy significativos."}
                  </p>
                  <p style={{ fontSize: '0.85rem' }}>
                    {"Phillips (1986) demostró formalmente que el t-estadístico diverge $\\to \\infty$ a medida que crece $T$. Imposible controlar el Error Tipo I."}
                  </p>
                </div>

                <div className="glass-card" style={{ borderTop: '3px solid var(--green)', padding: '1.25rem', margin: '0' }}>
                  <h4 style={{ color: 'var(--green)', marginBottom: '0.5rem', textTransform: 'uppercase', fontSize: '0.8rem', letterSpacing: '0.05em' }}>
                    LA SOLUCIÓN: Validar Estacionariedad
                  </h4>
                  <p style={{ fontSize: '0.85rem', marginBottom: '0.75rem' }}>
                    Verificar la estacionariedad de las series <strong>ANTES</strong> de estimar cualquier modelo econométrico.
                  </p>
                  <p style={{ fontSize: '0.85rem', marginBottom: '0.75rem' }}>
                    Para eso se necesita contrastar si la serie tiene raíz unitaria.
                  </p>
                  <p style={{ fontSize: '0.85rem' }}>
                    {"Dickey &amp; Fuller (1979) construyeron la herramienta estadística formal para hacerlo: desarrollaron el estadístico $\\hat{\\tau}$ y demostraron su distribución asintótica exacta."}
                  </p>
                </div>
              </div>
            </section>
          )}

          {activeSec === 'estacionariedad' && (
            <section className="theory-section" id="estacionariedad">
              <span className="section-tag">Diapositiva 3 de 9 — Proceso AR(1)</span>
              <h2>Estacionariedad y Proceso AR(1)</h2>
              <p style={{ fontSize: '0.85rem', marginBottom: '0.75rem' }}>
                Un proceso es estacionario en covarianza si cumple tres condiciones para todo $t$:
              </p>

              <div className="grid-3" style={{ gap: '0.75rem', marginBottom: '1rem' }}>
                <div className="grid-cell" style={{ padding: '0.75rem' }}>
                  <div className="cell-label" style={{ fontSize: '0.7rem' }}>Media constante</div>
                  <div className="cell-val" style={{ fontSize: '0.9rem', margin: '0.25rem 0' }}>{"$$E[y_t] = \\mu, \\quad \\forall t$$"}</div>
                </div>
                <div className="grid-cell" style={{ padding: '0.75rem' }}>
                  <div className="cell-label" style={{ fontSize: '0.7rem' }}>Varianza finita</div>
                  <div className="cell-val" style={{ fontSize: '0.9rem', margin: '0.25rem 0' }}>{"$$Var[y_t] = \\sigma^2 < \\infty, \\quad \\forall t$$"}</div>
                </div>
                <div className="grid-cell" style={{ padding: '0.75rem' }}>
                  <div className="cell-label" style={{ fontSize: '0.7rem' }}>Autocovarianza</div>
                  <div className="cell-val" style={{ fontSize: '0.9rem', margin: '0.25rem 0' }}>{"$$Cov(y_t, y_{t-k}) = \\gamma_k$$"}</div>
                </div>
              </div>

              <h4 style={{ color: 'var(--accent)', marginBottom: '0.5rem' }}>El Proceso AR(1)</h4>
              <div className="math-block" style={{ margin: '0.75rem 0', padding: '0.5rem' }}>
                {"$$y_t = \\rho y_{t-1} + \\epsilon_t, \\quad \\epsilon_t \\sim \\text{i.i.d.}(0, \\sigma^2)$$"}
              </div>

              <div className="grid-3" style={{ gap: '0.75rem', marginBottom: '0.75rem' }}>
                <div className="grid-cell" style={{ borderLeft: '3px solid var(--green)', padding: '0.5rem' }}>
                  <span className="cell-label" style={{ color: 'var(--green)', fontSize: '0.75rem', fontWeight: 'bold' }}>{"$|\\rho| < 1$"}</span>
                  <p style={{ fontSize: '0.75rem', margin: 0 }}>Proceso ESTACIONARIO</p>
                </div>
                <div className="grid-cell" style={{ borderLeft: '3px solid var(--warning)', padding: '0.5rem' }}>
                  <span className="cell-label" style={{ color: 'var(--warning)', fontSize: '0.75rem', fontWeight: 'bold' }}>{"$\\rho = 1$"}</span>
                  <p style={{ fontSize: '0.75rem', margin: 0 }}>RAÍZ UNITARIA (caminata aleatoria)</p>
                </div>
                <div className="grid-cell" style={{ borderLeft: '3px solid var(--danger)', padding: '0.5rem' }}>
                  <span className="cell-label" style={{ color: 'var(--danger)', fontSize: '0.75rem', fontWeight: 'bold' }}>{"$|\\rho| > 1$"}</span>
                  <p style={{ fontSize: '0.75rem', margin: 0 }}>Proceso NO ESTACIONARIO</p>
                </div>
              </div>

              <div className="glass-card" style={{ padding: '0.75rem', margin: '0' }}>
                <p style={{ fontSize: '0.8rem', margin: 0 }}>
                  {"Cuando $\\rho = 1$: la varianza crece sin límite"}
                </p>
                <div className="math-block" style={{ margin: '0.4rem 0', padding: '0.25rem' }}>
                  {"$$y_t = y_0 + \\sum_{i=1}^t \\epsilon_i \\quad \\implies \\quad Var[y_t] = t \\sigma^2 \\to \\infty$$"}
                </div>
                <p style={{ fontSize: '0.75rem', margin: 0, color: 'var(--text-muted)' }}>
                  (proceso integrado de orden 1, denotado $I(1)$)
                </p>
              </div>
            </section>
          )}

          {activeSec === 'formulacion' && (
            <section className="theory-section" id="formulacion">
              <span className="section-tag">Diapositiva 4 de 9 — Formulación</span>
              <h2>La Prueba DF: Formulación Original (Dickey &amp; Fuller, 1979)</h2>
              <p style={{ fontSize: '0.85rem', marginBottom: '0.75rem' }}>
                {"Transformación algebraica clave: restar $y_{t-1}$ a ambos lados del $AR(1)$"}
              </p>
              <div className="math-block" style={{ margin: '0.75rem 0', padding: '0.5rem' }}>
                <span className="math-label">De AR(1) $\to$ Ecuación de regresión auxiliar</span>
                {"$$y_t - y_{t-1} = (\\rho - 1) y_{t-1} + \\epsilon_t \\quad \\implies \\quad \\Delta y_t = \\phi y_{t-1} + \\epsilon_t \\quad \\text{donde } \\phi = \\rho - 1$$"}
              </div>

              <div className="grid-2" style={{ gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '0.75rem' }}>
                <div className="glass-card" style={{ padding: '1rem', margin: '0' }}>
                  <h4 style={{ color: 'var(--accent)', marginBottom: '0.4rem', fontSize: '0.8rem' }}>Hipótesis de la Prueba</h4>
                  <p style={{ fontSize: '0.8rem', margin: '0 0 0.3rem 0' }}>
                    {"• $H_0: \\phi = 0 \\quad (\\rho = 1)$"}
                    <br />
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Raíz unitaria $\to$ serie NO estacionaria</span>
                  </p>
                  <p style={{ fontSize: '0.8rem', margin: 0 }}>
                    {"• $H_1: \\phi < 0 \\quad (\\rho < 1)$"}
                    <br />
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Sin raíz unitaria $\to$ serie estacionaria</span>
                  </p>
                </div>

                <div className="glass-card" style={{ padding: '1rem', margin: '0' }}>
                  <h4 style={{ color: 'var(--accent2)', marginBottom: '0.4rem', fontSize: '0.8rem' }}>Estadístico de Prueba</h4>
                  <div className="math-block" style={{ margin: '0.4rem 0', padding: '0.4rem' }}>
                    {"$$\\hat{\\tau} = \\frac{\\hat{\\phi}}{SE(\\hat{\\phi})}$$"}
                  </div>
                  <p style={{ fontSize: '0.75rem', color: 'var(--danger)', margin: 0 }}>
                    {"⚠ Bajo $H_0$, $\\hat{\\tau}$ NO sigue una distribución t de Student"}
                  </p>
                </div>
              </div>
            </section>
          )}

          {activeSec === 'modelos' && (
            <section className="theory-section" id="modelos">
              <span className="section-tag">Diapositiva 5 de 9 — Los Tres Modelos</span>
              <h2>Los Tres Modelos de Dickey-Fuller</h2>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginTop: '1.25rem' }}>
                <p style={{ fontSize: '0.95rem', lineHeight: '1.6', color: 'var(--text-light)', margin: 0 }}>
                  Dickey y Fuller (1979, 1981) consideraron tres especificaciones según los componentes deterministas incluidos en la regresión:
                </p>

                <div className="glass-card" style={{ padding: '1.5rem', margin: '0', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <div style={{ padding: '0.5rem 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    {"$$\\Delta y_t = \\phi y_{t-1} + \\varepsilon_t \\quad (Modelo \\ 1)$$"}
                  </div>
                  <div style={{ padding: '0.5rem 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    {"$$\\Delta y_t = \\alpha + \\phi y_{t-1} + \\varepsilon_t \\quad (Modelo \\ 2)$$"}
                  </div>
                  <div style={{ padding: '0.5rem 0' }}>
                    {"$$\\Delta y_t = \\alpha + \\beta t + \\phi y_{t-1} + \\varepsilon_t \\quad (Modelo \\ 3)$$"}
                  </div>
                </div>

                <p style={{ fontSize: '0.9rem', lineHeight: '1.6', color: 'var(--text-muted)', margin: 0 }}>
                  Cada modelo genera una distribución asintótica diferente bajo $H_0$, implicando valores críticos distintos (Said & Dickey, 1984).
                </p>
              </div>
            </section>
          )}

          {activeSec === 'tendencia' && (
            <section className="theory-section" id="tendencia">
              <span className="section-tag">Diapositiva 6 de 9 — Modelo con Tendencia</span>
              <h2>Prueba de Dickey-Fuller con tendencia</h2>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginTop: '1.25rem' }}>
                <p style={{ fontSize: '0.95rem', lineHeight: '1.6', color: 'var(--text-light)', margin: 0 }}>
                  Cuando las series de tiempo tiene una tendencia marcada, la ecuación de la prueba de DF, añade el componente de tendencia, así:
                </p>

                <div className="glass-card" style={{ padding: '2rem', margin: '0', textAlign: 'center' }}>
                  {"$$\\Delta y_t = \\mu + \\delta T + \\theta y_{t-1} + e_t \\sim RB(0, \\sigma^2)$$"}
                </div>
              </div>
            </section>
          )}

          {activeSec === 'ampliada' && (
            <section className="theory-section" id="ampliada">
              <span className="section-tag">Diapositiva 7 de 9 — Prueba Ampliada</span>
              <h2>Prueba de Dickey Fuller Ampliada</h2>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginTop: '1.25rem' }}>
                <p style={{ fontSize: '0.95rem', lineHeight: '1.6', color: 'var(--text-light)', margin: 0 }}>
                  Esta prueba añade a la ecuación de la prueba de Dickey-Fuller los rezagos de $\Delta y_t$, así:
                </p>

                <div className="glass-card" style={{ padding: '2rem', margin: '0', textAlign: 'center' }}>
                  {"$$\\Delta y_t = \\mu + \\theta y_{t-1} + \\gamma \\Delta y_{t-1} + e_t \\sim RB(0, \\sigma^2)$$"}
                </div>
              </div>
            </section>
          )}

          {activeSec === 'estadistico' && (
            <section className="theory-section" id="estadistico">
              <span className="section-tag">Diapositiva 8 de 9 — Estadístico de Prueba</span>
              <h2>Estadístico de Prueba</h2>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginTop: '1.25rem' }}>
                <div className="glass-card" style={{ borderLeft: '3px solid var(--accent)', padding: '1.25rem', margin: '0' }}>
                  <p style={{ fontSize: '0.9rem', lineHeight: '1.6', margin: 0 }}>
                    El estadístico de prueba es el sobre la variable dependiente rezagada. Si el coeficiente de la variable dependiente rezagada será positivo. Si es igual a la unidad, En ambos casos será no estacionaria.
                  </p>
                </div>

                <div className="glass-card" style={{ borderLeft: '3px solid var(--accent2)', padding: '1.25rem', margin: '0' }}>
                  <p style={{ fontSize: '0.9rem', lineHeight: '1.6', margin: 0 }}>
                    Cuando existe tendencia en una serie temporal en un modelo AR (1), el primer regresor tenderá a ser 1 o muy cercano a 1. Esto se debe a la propiedad de reversión a la media de un proceso estocástico estacionario, es decir, cuanto más cerca esté el primer coeficiente de un modelo AR(1) de 1, más tardarán las observaciones a volver al valor medio.
                  </p>
                </div>
              </div>
            </section>
          )}

          {activeSec === 'criterio' && (
            <section className="theory-section" id="criterio">
              <span className="section-tag">Diapositiva 9 de 9 — Criterio de Decisión</span>
              <h2>Criterio de Decisión</h2>

              <div className="grid-2" style={{ gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginTop: '1.5rem' }}>
                <div className="glass-card" style={{ borderTop: '4px solid var(--danger)', padding: '1.5rem', margin: '0' }}>
                  <h4 style={{ color: 'var(--danger)', marginBottom: '0.75rem', textTransform: 'uppercase', fontSize: '0.85rem', letterSpacing: '0.05em' }}>
                    No rechazar la hipótesis nula (p-value &gt; 0.05)
                  </h4>
                  <p style={{ fontSize: '0.9rem', lineHeight: '1.6', margin: 0 }}>
                    Si no puede rechazarse la hipótesis nula, significa que (p-value &gt; 0.05), por tanto la serie es no estacionaria y tiene raíz 1 (I(1)). La serie es Random walk = no estacionaria.
                  </p>
                </div>

                <div className="glass-card" style={{ borderTop: '4px solid var(--green)', padding: '1.5rem', margin: '0' }}>
                  <h4 style={{ color: 'var(--green)', marginBottom: '0.75rem', textTransform: 'uppercase', fontSize: '0.85rem', letterSpacing: '0.05em' }}>
                    Rechazar la hipótesis nula (p-value &lt; 0.05)
                  </h4>
                  <p style={{ fontSize: '0.9rem', lineHeight: '1.6', margin: 0 }}>
                    si se rechaza la nula (p-valor&lt;0.05) la serie es estacionaria y tiene una raíz 0 (I(0)). La serie es White noise = estacionaria
                  </p>
                </div>
              </div>
            </section>
          )}

          {/* Navigation Controls inside the card */}
          <div className="steps-nav" style={{ marginTop: '2.5rem' }}>
            <button
              className="step-nav-btn"
              disabled={activeIndex === 0}
              onClick={() => setActiveSec(sections[activeIndex - 1].id)}
            >
              &larr; Diapositiva Anterior
            </button>
            <span className="steps-indicator" style={{ fontFamily: 'var(--heading)', fontWeight: '600' }}>
              Diapositiva {activeIndex + 1} de {sections.length}
            </span>
            <button
              className="step-nav-btn"
              disabled={activeIndex === sections.length - 1}
              onClick={() => setActiveSec(sections[activeIndex + 1].id)}
            >
              Siguiente Diapositiva &rarr;
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ============================================================================
   CALCULATOR TAB (INTERACTIVE REGRESSION CALCULATOR & MANUAL WALKTHROUGH)
   ============================================================================ */
function CalculatorTab() {
  const defaultSeries = [20, 22, 23, 25, 26, 28, 29, 31, 32, 34];
  const [series, setSeries] = useState(defaultSeries);
  const [currentStep, setCurrentStep] = useState(0);
  const containerRef = useRef(null);

  useEffect(() => {
    if (window.MathJax && window.MathJax.typesetPromise && containerRef.current) {
      window.MathJax.typesetClear([containerRef.current]);
      window.MathJax.typesetPromise([containerRef.current]).catch((err) =>
        console.error('MathJax error during render: ', err)
      );
    }
  }, [currentStep, series]);

  const stats = useMemo(() => calculateDF(series), [series]);

  const handleInputChange = (index, val) => {
    const next = [...series];
    const parsed = parseFloat(val);
    next[index] = isNaN(parsed) ? 0 : parsed;
    setSeries(next);
  };

  const handleReset = () => {
    setSeries(defaultSeries);
    setCurrentStep(0);
  };

  // SVGs Charting helpers for original series
  const seriesMinMax = useMemo(() => {
    const min = Math.min(...series);
    const max = Math.max(...series);
    const range = max - min;
    return {
      min: min - (range === 0 ? 1 : range * 0.1),
      max: max + (range === 0 ? 1 : range * 0.1)
    };
  }, [series]);

  const mapPointToSvg = (index, value, width, height, padding) => {
    const xRange = series.length - 1;
    const yRange = seriesMinMax.max - seriesMinMax.min;

    const x = padding.left + (index / xRange) * (width - padding.left - padding.right);
    const y = padding.top + ((seriesMinMax.max - value) / yRange) * (height - padding.top - padding.bottom);
    return { x, y };
  };

  const originalSeriesSvgPath = useMemo(() => {
    const width = 450;
    const height = 200;
    const padding = { top: 15, right: 15, bottom: 20, left: 35 };

    const points = series.map((val, idx) => mapPointToSvg(idx, val, width, height, padding));
    return points.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
  }, [series, seriesMinMax]);

  // SVG Charting for Regression scatter & line
  const regressionMinMax = useMemo(() => {
    if (!stats) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    const minX = Math.min(...stats.X);
    const maxX = Math.max(...stats.X);
    const minY = Math.min(...stats.Y);
    const maxY = Math.max(...stats.Y);
    const rangeX = maxX - minX;
    const rangeY = maxY - minY;
    return {
      minX: minX - (rangeX === 0 ? 1 : rangeX * 0.15),
      maxX: maxX + (rangeX === 0 ? 1 : rangeX * 0.15),
      minY: minY - (rangeY === 0 ? 1 : rangeY * 0.15),
      maxY: maxY + (rangeY === 0 ? 1 : rangeY * 0.15)
    };
  }, [stats]);

  const mapRegressionToSvg = (xVal, yVal, width, height, padding) => {
    const { minX, maxX, minY, maxY } = regressionMinMax;
    const xRange = maxX - minX;
    const yRange = maxY - minY;

    const x = padding.left + ((xVal - minX) / xRange) * (width - padding.left - padding.right);
    const y = padding.top + ((maxY - yVal) / yRange) * (height - padding.top - padding.bottom);
    return { x, y };
  };

  const regressionLineSvgPath = useMemo(() => {
    if (!stats) return '';
    const width = 450;
    const height = 200;
    const padding = { top: 15, right: 15, bottom: 20, left: 35 };
    const { minX, maxX } = regressionMinMax;

    const yStart = stats.beta0 + stats.beta1 * minX;
    const yEnd = stats.beta0 + stats.beta1 * maxX;

    const pStart = mapRegressionToSvg(minX, yStart, width, height, padding);
    const pEnd = mapRegressionToSvg(maxX, yEnd, width, height, padding);

    return `M ${pStart.x} ${pStart.y} L ${pEnd.x} ${pEnd.y}`;
  }, [stats, regressionMinMax]);

  const isRejected = stats && stats.t_stat < -2.86;

  return (
    <div className="tab-panel" ref={containerRef}>
      <h2>Calculadora Paso a Paso con Datos Dinámicos</h2>
      <p style={{ fontFamily: 'var(--serif)', fontStyle: 'italic', marginBottom: '1.5rem' }}>
        Modifica los valores del panel izquierdo. Todos los cálculos matemáticos, tablas, gráficos de regresión y decisiones se recalcularán automáticamente en tiempo real.
      </p>

      <div className="calculator-grid">
        {/* Panel Izquierdo: Inputs */}
        <aside className="glass-card series-input-card">
          <div className="input-header">
            <h3>{"Serie Temporal ($Y_t$)"}</h3>
            <button className="reset-btn" onClick={handleReset}>
              Restablecer
            </button>
          </div>
          <div className="data-grid-inputs">
            {series.map((val, idx) => (
              <div className="input-cell" key={idx}>
                <span className="input-index">t = {idx + 1}</span>
                <input
                  className="input-field"
                  type="number"
                  step="any"
                  value={val}
                  onChange={(e) => handleInputChange(idx, e.target.value)}
                />
              </div>
            ))}
          </div>
        </aside>

        {/* Panel Derecho: Gráficos y Pasos */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Gráficos */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1rem' }}>
            {/* Chart original */}
            <div className="chart-container">
              <div className="chart-title">
                <span>{"Serie Temporal ($Y_t$)"}</span>
                <div className="chart-legend">
                  <div className="legend-item">
                    <div className="legend-color" style={{ background: 'var(--accent)' }}></div>
                    <span>{"Ventas ($Y_t$)"}</span>
                  </div>
                </div>
              </div>
              <svg className="svg-chart" viewBox="0 0 450 200">
                {/* Gridlines */}
                {[0, 0.25, 0.5, 0.75, 1].map((p, i) => {
                  const yVal = seriesMinMax.min + p * (seriesMinMax.max - seriesMinMax.min);
                  const pt = mapPointToSvg(0, yVal, 450, 200, { top: 15, right: 15, bottom: 20, left: 35 });
                  return (
                    <g key={i}>
                      <line className="chart-grid-line" x1="35" y1={pt.y} x2="435" y2={pt.y} />
                      <text className="chart-axis-text" x="5" y={pt.y + 3}>{yVal.toFixed(1)}</text>
                    </g>
                  );
                })}
                {/* Horizontal Axis */}
                <line className="chart-axis-line" x1="35" y1="180" x2="435" y2="180" />
                {/* Labels X */}
                {series.map((_, idx) => {
                  const pt = mapPointToSvg(idx, seriesMinMax.min, 450, 200, { top: 15, right: 15, bottom: 20, left: 35 });
                  return (
                    <text key={idx} className="chart-axis-text" x={pt.x - 3} y="195">{idx + 1}</text>
                  );
                })}
                {/* Path line */}
                <path className="series-line" d={originalSeriesSvgPath} />
                {/* Points */}
                {series.map((val, idx) => {
                  const pt = mapPointToSvg(idx, val, 450, 200, { top: 15, right: 15, bottom: 20, left: 35 });
                  return (
                    <circle className="chart-dot" cx={pt.x} cy={pt.y} r="4.5" key={idx} />
                  );
                })}
              </svg>
            </div>

            {/* Chart Regresión */}
            <div className="chart-container">
              <div className="chart-title">
                <span>{"Regresión: $\\Delta Y_t$ vs $Y_{t-1}$"}</span>
                <div className="chart-legend">
                  <div className="legend-item">
                    <div className="legend-color" style={{ background: 'var(--warning)', height: '2px', borderTop: '2px dashed var(--warning)' }}></div>
                    <span>Recta MCO</span>
                  </div>
                </div>
              </div>
              {stats ? (
                <svg className="svg-chart" viewBox="0 0 450 200">
                  {/* Grid Y */}
                  {[0, 0.25, 0.5, 0.75, 1].map((p, i) => {
                    const yVal = regressionMinMax.minY + p * (regressionMinMax.maxY - regressionMinMax.minY);
                    const pt = mapRegressionToSvg(regressionMinMax.minX, yVal, 450, 200, { top: 15, right: 15, bottom: 20, left: 35 });
                    return (
                      <g key={i}>
                        <line className="chart-grid-line" x1="35" y1={pt.y} x2="435" y2={pt.y} />
                        <text className="chart-axis-text" x="5" y={pt.y + 3}>{yVal.toFixed(2)}</text>
                      </g>
                    );
                  })}
                  {/* Horizontal Axis (y=0 line if in range) */}
                  {0 >= regressionMinMax.minY && 0 <= regressionMinMax.maxY && (
                    <line className="chart-axis-line" style={{ stroke: 'rgba(255,255,255,0.06)' }} x1="35" y1={mapRegressionToSvg(regressionMinMax.minX, 0, 450, 200, { top: 15, right: 15, bottom: 20, left: 35 }).y} x2="435" y2={mapRegressionToSvg(regressionMinMax.minX, 0, 450, 200, { top: 15, right: 15, bottom: 20, left: 35 }).y} />
                  )}
                  {/* Scatter & Line */}
                  <path className="regression-line" d={regressionLineSvgPath} />
                  {stats.X.map((xVal, idx) => {
                    const yVal = stats.Y[idx];
                    const pt = mapRegressionToSvg(xVal, yVal, 450, 200, { top: 15, right: 15, bottom: 20, left: 35 });
                    return (
                      <circle className="chart-dot regression" cx={pt.x} cy={pt.y} r="5.5" key={idx} />
                    );
                  })}
                </svg>
              ) : (
                <div style={{ height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  Datos insuficientes.
                </div>
              )}
            </div>
          </div>

          {/* Pasos explicativos */}
          {stats && (
            <div className="walkthrough-container">
              <div className="steps-nav">
                <button
                  className="step-nav-btn"
                  disabled={currentStep === 0}
                  onClick={() => setCurrentStep((s) => s - 1)}
                >
                  &larr; Anterior
                </button>
                <span className="steps-indicator">Paso {currentStep + 1} de 6</span>
                <button
                  className="step-nav-btn"
                  disabled={currentStep === 5}
                  onClick={() => setCurrentStep((s) => s + 1)}
                >
                  Siguiente &rarr;
                </button>
              </div>

              <div className="step-slide-container">
                {currentStep === 0 && (
                  <div className="glass-card step-card" key="step0">
                    <div className="step-title-row">
                      <div className="step-number-badge">1</div>
                      <h4>Plantear las Hipótesis de Contraste</h4>
                    </div>
                    <div className="step-body">
                      <p>
                        {"Para determinar si nuestra serie de ventas mensuales es estacionaria (estable) o sigue un camino aleatorio no estacionario, contrastamos el coeficiente de retraso $\\phi = \\rho - 1$ en la regresión auxiliar de diferencias:"}
                      </p>
                      <div className="grid-3" style={{ gridTemplateColumns: '1fr 1fr', gap: '1.5rem', margin: '1rem 0' }}>
                        <div className="grid-cell" style={{ borderLeft: '4px solid var(--danger)', padding: '1rem 1.25rem' }}>
                          <div className="cell-label" style={{ color: 'var(--danger)' }}>H0 - Hipótesis Nula</div>
                          <p style={{ fontWeight: '500', color: 'var(--text-white)' }}>{"$$\\phi = 0 \\quad (\\rho = 1)$$"}</p>
                          <p style={{ fontSize: '0.8rem', marginTop: '0.35rem' }}>La serie posee raíz unitaria y <strong>NO es estacionaria</strong>. Los impactos en la serie son permanentes.</p>
                        </div>
                        <div className="grid-cell" style={{ borderLeft: '4px solid var(--green)', padding: '1rem 1.25rem' }}>
                          <div className="cell-label" style={{ color: 'var(--green)' }}>Ha - Hipótesis Alternativa</div>
                          <p style={{ fontWeight: '500', color: 'var(--text-white)' }}>{"$$\\phi < 0 \\quad (\\rho < 1)$$"}</p>
                          <p style={{ fontSize: '0.8rem', marginTop: '0.35rem' }}>La serie es <strong>estacionaria</strong> y tiende a regresar a su media de largo plazo. Los shocks son transitorios.</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {currentStep === 1 && (
                  <div className="glass-card step-card" key="step1">
                    <div className="step-title-row">
                      <div className="step-number-badge">2</div>
                      <h4>Calcular Diferencias y Medias Muestrales</h4>
                    </div>
                    <div className="step-body">
                      <p>
                        {"La regresión auxiliar se define en términos de diferencias $\\Delta Y_t = Y_t - Y_{t-1}$ como variable dependiente, y el retardo $Y_{t-1}$ como regresor."}
                      </p>
                      <p>
                        {`Con $m = ${series.length}$ observaciones originales, obtenemos $n = ${stats.n}$ observaciones útiles de regresión. Definimos:`}
                      </p>
                      <div className="math-block">
                        {"$$X_i = Y_{t-1} \\qquad Y_i = \\Delta Y_t$$"}
                      </div>
                      <p>Calculamos las medias aritméticas de ambas series:</p>
                      <div className="grid-3" style={{ gridTemplateColumns: '1fr 1fr', margin: '1rem 0' }}>
                        <div className="grid-cell">
                          <div className="cell-label">{"Media de X ($\\bar{X}$)"}</div>
                          <div className="cell-val" style={{ fontSize: '1.25rem', color: 'var(--accent)' }}>
                            {`$$\\bar{X} = \\frac{\\sum x_i}{n} = \\frac{${stats.sum_X.toFixed(2)}}{${stats.n}} \\approx ${stats.mean_X.toFixed(4)}$$`}
                          </div>
                        </div>
                        <div className="grid-cell">
                          <div className="cell-label">{"Media de Y ($\\bar{Y}$)"}</div>
                          <div className="cell-val" style={{ fontSize: '1.25rem', color: 'var(--accent2)' }}>
                            {`$$\\bar{Y} = \\frac{\\sum y_i}{n} = \\frac{${stats.sum_Y.toFixed(2)}}{${stats.n}} \\approx ${stats.mean_Y.toFixed(4)}$$`}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {currentStep === 2 && (
                  <div className="glass-card step-card" key="step2">
                    <div className="step-title-row">
                      <div className="step-number-badge">3</div>
                      <h4>Construcción de la Tabla de Trabajo Auxiliar</h4>
                    </div>
                    <div className="step-body">
                      <p>
                        {"Para resolver por MCO, construimos la tabla con los desvíos respecto a las medias, sus productos cruzados y la suma de cuadrados del regresor:"}
                      </p>
                      <div className="premium-table-container">
                        <table className="premium-table" style={{ fontSize: '0.75rem' }}>
                          <thead>
                            <tr>
                              <th>t</th>
                              <th>{"$Y_t$"}</th>
                              <th className="highlight-header-x">{"$X_i = Y_{t-1}$"}</th>
                              <th className="highlight-header-y">{"$Y_i = \\Delta Y_t$"}</th>
                              <th className="highlight-header-x">{"$X_i - \\bar{X}$"}</th>
                              <th className="highlight-header-y">{"$Y_i - \\bar{Y}$"}</th>
                              <th className="highlight-header-prod">Prod. Desv.</th>
                              <th className="highlight-header-sq">{"Desv. Cuad. ($X_i - \\bar{X})^2$"}</th>
                            </tr>
                          </thead>
                          <tbody>
                            {series.map((val, idx) => {
                              if (idx === 0) {
                                return (
                                  <tr key={idx}>
                                    <td>1</td>
                                    <td>{val.toFixed(1)}</td>
                                    <td className="highlight-x">-</td>
                                    <td className="highlight-y">-</td>
                                    <td className="highlight-x">-</td>
                                    <td className="highlight-y">-</td>
                                    <td className="highlight-prod">-</td>
                                    <td className="highlight-sq">-</td>
                                  </tr>
                                );
                              }
                              const xi = stats.X[idx - 1];
                              const yi = stats.Y[idx - 1];
                              const x_diff = stats.X_diff[idx - 1];
                              const y_diff = stats.Y_diff[idx - 1];
                              const prod = stats.prod[idx - 1];
                              const sq = stats.sq[idx - 1];
                              return (
                                <tr key={idx}>
                                  <td>{idx + 1}</td>
                                  <td>{val.toFixed(1)}</td>
                                  <td className="highlight-x">{Number.isInteger(xi) ? xi.toFixed(0) : xi.toFixed(2)}</td>
                                  <td className="highlight-y">{Number.isInteger(yi) ? yi.toFixed(0) : yi.toFixed(2)}</td>
                                  <td className="highlight-x">{x_diff.toFixed(2)}</td>
                                  <td className="highlight-y">{y_diff.toFixed(2)}</td>
                                  <td className="highlight-prod">{prod.toFixed(2)}</td>
                                  <td className="highlight-sq">{sq.toFixed(2)}</td>
                                </tr>
                              );
                            })}
                            <tr style={{ fontWeight: '700', borderTop: '2px solid var(--accent)' }}>
                              <td colSpan="2">Medias / Sumas</td>
                              <td className="highlight-x" style={{ color: 'var(--accent)' }}>{stats.mean_X.toFixed(2)}</td>
                              <td className="highlight-y" style={{ color: 'var(--accent2)' }}>{stats.mean_Y.toFixed(2)}</td>
                              <td className="highlight-x">-</td>
                              <td className="highlight-y">-</td>
                              <td className="highlight-prod" style={{ color: 'var(--warning)', fontWeight: 'bold' }}>{stats.sum_prod.toFixed(2)}</td>
                              <td className="highlight-sq" style={{ color: 'var(--green)', fontWeight: 'bold' }}>{stats.sum_sq.toFixed(2)}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}

                {currentStep === 3 && (
                  <div className="glass-card step-card" key="step3">
                    <div className="step-title-row">
                      <div className="step-number-badge">4</div>
                      <h4>{"Calcular Coeficientes MCO ($\\hat{\\beta}_1, \\hat{\\beta}_0$)"}</h4>
                    </div>
                    <div className="step-body">
                      <p>
                        {"Estimamos la ecuación de regresión lineal $\\Delta Y_t = \\hat{\\beta}_0 + \\hat{\\beta}_1 Y_{t-1}$."}
                      </p>
                      <p>{"La pendiente $\\hat{\\beta}_1$ representa nuestra tasa de decaimiento temporal y se define como:"}</p>
                      <div className="math-block">
                        {`$$\\hat{\\beta}_1 = \\frac{\\sum (X_i - \\bar{X})(Y_i - \\bar{Y})}{\\sum (X_i - \\bar{X})^2} = \\frac{${stats.sum_prod.toFixed(4)}}{${stats.sum_sq.toFixed(4)}} \\approx ${stats.beta1.toFixed(6)}$$`}
                      </div>
                      <p>{"El intercepto $\\hat{\\beta}_0$ (que representa la deriva de la serie $\\alpha$):"}</p>
                      <div className="math-block">
                        {`$$\\hat{\\beta}_0 = \\bar{Y} - \\hat{\\beta}_1 \\bar{X} = ${stats.mean_Y.toFixed(4)} - (${stats.beta1.toFixed(5)} \\cdot ${stats.mean_X.toFixed(4)}) \\approx ${stats.beta0.toFixed(6)}$$`}
                      </div>
                      <p>La ecuación auxiliar estimada resulta en:</p>
                      <div className="math-block" style={{ color: 'var(--warning)', fontWeight: '600' }}>
                        {`$$\\widehat{\\Delta Y_t} = ${stats.beta0.toFixed(4)} ${stats.beta1 >= 0 ? '+' : ''} ${stats.beta1.toFixed(4)} \\cdot Y_{t-1}$$`}
                      </div>
                    </div>
                  </div>
                )}

                {currentStep === 4 && (
                  <div className="glass-card step-card" key="step4">
                    <div className="step-title-row">
                      <div className="step-number-badge">5</div>
                      <h4>Calcular Residuos, Varianza y Error Estándar</h4>
                    </div>
                    <div className="step-body">
                      <p>
                        Para inferir sobre el coeficiente de la pendiente, necesitamos calcular la varianza de los residuos y el error estándar de la estimación.
                      </p>
                      <p>
                        {"1. La Suma de Cuadrados de Residuos ($SSR = \\sum e_i^2$ con $e_i = Y_i - \\hat{Y}_i$):"}
                      </p>
                      <div className="math-block">
                        {`$$SSR \\approx ${stats.sum_resids_sq.toFixed(4)}$$`}
                      </div>
                      <p>
                        {"2. Varianza de los residuos ($\\hat{\\sigma}^2$) con $n - 2$"} = {stats.df} grados de libertad:
                      </p>
                      <div className="math-block">
                        {`$$\\hat{\\sigma}^2 = \\frac{SSR}{n - 2} = \\frac{${stats.sum_resids_sq.toFixed(4)}}{${stats.df}} \\approx ${stats.sigma_sq.toFixed(6)}$$`}
                      </div>
                      <p>
                        {"3. Varianza teórica de la pendiente estimada $Var(\\hat{\\beta}_1)$:"}
                      </p>
                      <div className="math-block">
                        {`$$Var(\\hat{\\beta}_1) = \\frac{\\hat{\\sigma}^2}{\\sum (X_i - \\bar{X})^2} = \\frac{${stats.sigma_sq.toFixed(5)}}{${stats.sum_sq.toFixed(4)}} \\approx ${stats.var_beta1.toFixed(8)}$$`}
                      </div>
                      <p>
                        {"4. Error Estándar del coeficiente $ES(\\hat{\\beta}_1)$:"}
                      </p>
                      <div className="math-block">
                        {`$$ES(\\hat{\\beta}_1) = \\sqrt{Var(\\hat{\\beta}_1)} = \\sqrt{${stats.var_beta1.toFixed(8)}} \\approx ${stats.se_beta1.toFixed(6)}$$`}
                      </div>
                    </div>
                  </div>
                )}

                {currentStep === 5 && (
                  <div className="glass-card step-card" key="step5">
                    <div className="step-title-row">
                      <div className="step-number-badge">6</div>
                      <h4>Estadístico de Contraste y Decisión Estadística</h4>
                    </div>
                    <div className="step-body">
                      <p>
                        {"El estadístico de Dickey-Fuller ($\\tau$) es la razón t-Student de la pendiente:"}
                      </p>
                      <div className="math-block">
                        {`$$\\tau = \\frac{\\hat{\\beta}_1}{ES(\\hat{\\beta}_1)} = \\frac{${stats.beta1.toFixed(6)}}{${stats.se_beta1.toFixed(6)}} \\approx ${stats.t_stat.toFixed(4)}$$`}
                      </div>
                      <p>
                        El valor crítico tabulado al <strong>5% de significancia</strong> para la regresión con constante (Modelo 2, muestra pequeña) es <strong>-2.86</strong>.
                      </p>

                      <div className="premium-table-container" style={{ maxWidth: '350px', margin: '1rem auto' }}>
                        <table className="premium-table">
                          <thead>
                            <tr>
                              <th>Nivel de Signif.</th>
                              <th>Valor Crítico DF</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              <td>1%</td>
                              <td>-3.43</td>
                            </tr>
                            <tr style={{ background: 'rgba(245, 158, 11, 0.08)', fontWeight: 'bold' }}>
                              <td>5% &larr;</td>
                              <td style={{ color: 'var(--warning)' }}>-2.86</td>
                            </tr>
                            <tr>
                              <td>10%</td>
                              <td>-2.57</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>

                      {isRejected ? (
                        <div className="decision-card reject">
                          <div className="decision-icon">
                            <ShieldIcon isGreen={true} />
                          </div>
                          <div className="decision-text">
                            <div className="decision-badge-lbl">Hipótesis Nula Rechazada</div>
                            <h5 className="decision-title">Se rechaza H0: La Serie es Estacionaria</h5>
                            <p className="decision-desc">
                              {`El estadístico $\\tau = ${stats.t_stat.toFixed(4)}$ es **más negativo** que el valor crítico de $-2.86$. Hay suficiente evidencia estadística para rechazar la presencia de raíz unitaria. La serie es estacionaria en covarianza (tipo $I(0)$).`}
                            </p>
                          </div>
                        </div>
                      ) : (
                        <div className="decision-card fail-reject">
                          <div className="decision-icon">
                            <ShieldIcon isGreen={false} />
                          </div>
                          <div className="decision-text">
                            <div className="decision-badge-lbl">No se Rechaza H0</div>
                            <h5 className="decision-title">No se rechaza H0: Serie No Estacionaria (Raíz Unitaria)</h5>
                            <p className="decision-desc">
                              {`El estadístico $\\tau = ${stats.t_stat.toFixed(4)}$ es **mayor (menos negativo)** que el valor crítico de $-2.86$. No disponemos de evidencia suficiente para rechazar la hipótesis nula. La serie temporal es no estacionaria (contiene una raíz unitaria, $I(1)$).`}
                            </p>
                          </div>
                        </div>
                      )}

                      <h4 style={{ marginTop: '2.5rem', marginBottom: '1rem', color: 'var(--text-white)' }}>Tabla de Trabajo Completa con Residuos</h4>
                      <p style={{ fontSize: '0.85rem', marginBottom: '1rem', color: 'var(--text-muted)' }}>
                        Esta es la tabla final consolidada, incluyendo los valores predichos y los residuos calculados a partir de los coeficientes estimados:
                      </p>
                      <div className="premium-table-container">
                        <table className="premium-table" style={{ fontSize: '0.75rem' }}>
                          <thead>
                            <tr>
                              <th>t</th>
                              <th>{"$Y_t$"}</th>
                              <th className="highlight-header-x">{"$X_i = Y_{t-1}$"}</th>
                              <th className="highlight-header-y">{"$Y_i = \\Delta Y_t$"}</th>
                              <th className="highlight-header-x">{"$X_i - \\bar{X}$"}</th>
                              <th className="highlight-header-y">{"$Y_i - \\bar{Y}$"}</th>
                              <th className="highlight-header-prod">Prod. Desv.</th>
                              <th className="highlight-header-sq">{"Desv. Cuad. ($X_i - \\bar{X})^2$"}</th>
                              <th className="highlight-header-y">{"$\\hat{Y}_i$"}</th>
                              <th>{"$\\hat{\\mu}_i$"}</th>
                              <th className="highlight-header-sq">{"$\\hat{\\mu}_i^2$"}</th>
                            </tr>
                          </thead>
                          <tbody>
                            {series.map((val, idx) => {
                              if (idx === 0) {
                                return (
                                  <tr key={idx}>
                                    <td>1</td>
                                    <td>{val.toFixed(1)}</td>
                                    <td className="highlight-x">-</td>
                                    <td className="highlight-y">-</td>
                                    <td className="highlight-x">-</td>
                                    <td className="highlight-y">-</td>
                                    <td className="highlight-prod">-</td>
                                    <td className="highlight-sq">-</td>
                                    <td className="highlight-y">-</td>
                                    <td>-</td>
                                    <td className="highlight-sq">-</td>
                                  </tr>
                                );
                              }
                              const xi = stats.X[idx - 1];
                              const yi = stats.Y[idx - 1];
                              const x_diff = stats.X_diff[idx - 1];
                              const y_diff = stats.Y_diff[idx - 1];
                              const prod = stats.prod[idx - 1];
                              const sq = stats.sq[idx - 1];
                              const y_pred = stats.Y_pred[idx - 1];
                              const resid = stats.resids[idx - 1];
                              const resid_sq = stats.resids_sq[idx - 1];
                              return (
                                <tr key={idx}>
                                  <td>{idx + 1}</td>
                                  <td>{val.toFixed(1)}</td>
                                  <td className="highlight-x">{Number.isInteger(xi) ? xi.toFixed(0) : xi.toFixed(2)}</td>
                                  <td className="highlight-y">{Number.isInteger(yi) ? yi.toFixed(0) : yi.toFixed(2)}</td>
                                  <td className="highlight-x">{x_diff.toFixed(2)}</td>
                                  <td className="highlight-y">{y_diff.toFixed(2)}</td>
                                  <td className="highlight-prod">{prod.toFixed(2)}</td>
                                  <td className="highlight-sq">{sq.toFixed(2)}</td>
                                  <td className="highlight-y">{y_pred.toFixed(2)}</td>
                                  <td>{resid.toFixed(2)}</td>
                                  <td className="highlight-sq">{resid_sq.toFixed(2)}</td>
                                </tr>
                              );
                            })}
                            <tr style={{ fontWeight: '700', borderTop: '2px solid var(--accent)' }}>
                              <td colSpan="2">Medias / Sumas</td>
                              <td className="highlight-x" style={{ color: 'var(--accent)' }}>{stats.mean_X.toFixed(2)}</td>
                              <td className="highlight-y" style={{ color: 'var(--accent2)' }}>{stats.mean_Y.toFixed(2)}</td>
                              <td className="highlight-x">-</td>
                              <td className="highlight-y">-</td>
                              <td className="highlight-prod" style={{ color: 'var(--warning)', fontWeight: 'bold' }}>{stats.sum_prod.toFixed(2)}</td>
                              <td className="highlight-sq" style={{ color: 'var(--green)', fontWeight: 'bold' }}>{stats.sum_sq.toFixed(2)}</td>
                              <td className="highlight-y">-</td>
                              <td>-</td>
                              <td className="highlight-sq" style={{ color: 'var(--green)', fontWeight: 'bold' }}>{stats.sum_resids_sq.toFixed(2)}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ============================================================================
   SIMULATION TAB (MONTE CARLO AND STOCHASTIC SIMULATOR)
   ============================================================================ */
function SimulationTab() {
  const [params, setParams] = useState({
    T: 40,
    rho: 1.0,
    drift: 0.5,
    trend: 0.0,
    noise: 1.0
  });

  const [seriesData, setSeriesData] = useState([]);
  const [rejectCount, setRejectCount] = useState(0);
  const [totalSims, setTotalSims] = useState(0);
  const containerRef = useRef(null);

  useEffect(() => {
    if (window.MathJax && window.MathJax.typesetPromise && containerRef.current) {
      window.MathJax.typesetClear([containerRef.current]);
      window.MathJax.typesetPromise([containerRef.current]).catch((err) =>
        console.error('MathJax error during render: ', err)
      );
    }
  }, [seriesData, totalSims]);

  // Generate series and run test
  const handleGenerate = () => {
    const data = generateAR1(params.T, params.rho, params.drift, params.trend, params.noise);
    setSeriesData(data);
  };

  // Run 100 simulations to compute power/rejection rate
  const handleRunMonteCarlo = () => {
    let rejections = 0;
    const size = 100;
    for (let s = 0; s < size; s++) {
      const data = generateAR1(params.T, params.rho, params.drift, params.trend, params.noise);
      const res = calculateDF(data);
      if (res && res.t_stat < -2.86) {
        rejections++;
      }
    }
    setRejectCount(rejections);
    setTotalSims(size);
    // set current view to the last generated path
    const lastData = generateAR1(params.T, params.rho, params.drift, params.trend, params.noise);
    setSeriesData(lastData);
  };

  useEffect(() => {
    handleGenerate();
    setTotalSims(0);
    setRejectCount(0);
  }, [params]);

  const stats = useMemo(() => calculateDF(seriesData), [seriesData]);

  // Scaler helpers for simulation chart
  const simMinMax = useMemo(() => {
    if (seriesData.length === 0) return { min: 0, max: 1 };
    const min = Math.min(...seriesData);
    const max = Math.max(...seriesData);
    const range = max - min;
    return {
      min: min - (range === 0 ? 1 : range * 0.1),
      max: max + (range === 0 ? 1 : range * 0.1)
    };
  }, [seriesData]);

  const mapSimToSvg = (index, value, width, height, padding) => {
    const xRange = params.T - 1;
    const yRange = simMinMax.max - simMinMax.min;

    const x = padding.left + (index / xRange) * (width - padding.left - padding.right);
    const y = padding.top + ((simMinMax.max - value) / yRange) * (height - padding.top - padding.bottom);
    return { x, y };
  };

  const simSvgPath = useMemo(() => {
    const width = 800;
    const height = 250;
    const padding = { top: 20, right: 20, bottom: 25, left: 45 };

    const points = seriesData.map((val, idx) => mapSimToSvg(idx, val, width, height, padding));
    return points.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
  }, [seriesData, simMinMax, params.T]);

  const isRejected = stats && stats.t_stat < -2.86;

  return (
    <div className="tab-panel" ref={containerRef}>
      <h2>Simulador de Procesos Estocásticos y Potencia de la Prueba</h2>
      <p style={{ fontFamily: 'var(--serif)', fontStyle: 'italic', marginBottom: '1.5rem' }}>
        {"Experimenta cómo el parámetro autoregresivo $\\rho$, la deriva ($\\alpha$) y la tendencia determinista afectan las trayectorias de la serie temporal y altera la capacidad de la prueba para detectar estacionariedad."}
      </p>

      {/* Grid de Controles */}
      <div className="glass-card">
        <div className="sim-controls">
          <div className="control-group">
            <div className="control-lbl-row">
              <span className="control-lbl">{"Coeficiente Autoregresivo ($\\rho$)"}</span>
              <span className="control-val">{params.rho.toFixed(2)}</span>
            </div>
            <input
              type="range"
              className="control-slider"
              min="0.0"
              max="1.1"
              step="0.05"
              value={params.rho}
              onChange={(e) => setParams({ ...params, rho: parseFloat(e.target.value) })}
            />
            <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
              {params.rho < 1
                ? 'Proceso Estacionario'
                : params.rho === 1
                  ? 'Caminata Aleatoria (I(1))'
                  : 'Proceso NO ESTACIONARIO'}
            </span>
          </div>

          <div className="control-group">
            <div className="control-lbl-row">
              <span className="control-lbl">{"Tamaño de Muestra ($T$)"}</span>
              <span className="control-val">{params.T}</span>
            </div>
            <input
              type="range"
              className="control-slider"
              min="20"
              max="100"
              step="5"
              value={params.T}
              onChange={(e) => setParams({ ...params, T: parseInt(e.target.value, 10) })}
            />
            <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Muestras más grandes aumentan la potencia.</span>
          </div>

          <div className="control-group">
            <div className="control-lbl-row">
              <span className="control-lbl">{"Constante / Deriva ($\\alpha$)"}</span>
              <span className="control-val">{params.drift.toFixed(1)}</span>
            </div>
            <input
              type="range"
              className="control-slider"
              min="0.0"
              max="5.0"
              step="0.5"
              value={params.drift}
              onChange={(e) => setParams({ ...params, drift: parseFloat(e.target.value) })}
            />
          </div>

          <div className="control-group">
            <div className="control-lbl-row">
              <span className="control-lbl">{"Desviación del Ruido ($\\sigma$)"}</span>
              <span className="control-val">{params.noise.toFixed(1)}</span>
            </div>
            <input
              type="range"
              className="control-slider"
              min="0.2"
              max="2.5"
              step="0.1"
              value={params.noise}
              onChange={(e) => setParams({ ...params, noise: parseFloat(e.target.value) })}
            />
          </div>
        </div>

        <div className="sim-actions">
          <button className="primary-btn" onClick={handleGenerate}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '2px' }}>
              <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67" />
            </svg>
            Generar Trayectoria
          </button>
          <button className="secondary-btn" onClick={handleRunMonteCarlo}>
            Ejecutar Monte Carlo (100 sim)
          </button>
        </div>
      </div>

      {/* Gráfico de la simulación */}
      <div className="chart-container" style={{ margin: '1.5rem 0' }}>
        <div className="chart-title">
          <span>{"Serie Temporal Simulada ($y_t$)"}</span>
          <div className="chart-legend">
            <div className="legend-item">
              <div className="legend-color" style={{ background: 'var(--accent)' }}></div>
              <span>Trayectoria AR(1)</span>
            </div>
          </div>
        </div>
        <svg className="svg-chart" style={{ height: '250px' }} viewBox="0 0 800 250">
          {/* Grid Y */}
          {[0, 0.25, 0.5, 0.75, 1].map((p, i) => {
            const yVal = simMinMax.min + p * (simMinMax.max - simMinMax.min);
            const pt = mapSimToSvg(0, yVal, 800, 250, { top: 20, right: 20, bottom: 25, left: 45 });
            return (
              <g key={i}>
                <line className="chart-grid-line" x1="45" y1={pt.y} x2="780" y2={pt.y} />
                <text className="chart-axis-text" x="5" y={pt.y + 3}>{yVal.toFixed(2)}</text>
              </g>
            );
          })}
          {/* Axis X */}
          <line className="chart-axis-line" x1="45" y1="225" x2="780" y2="225" />
          {/* Axis X ticks */}
          {Array.from({ length: 11 }).map((_, i) => {
            const index = Math.round((i / 10) * (params.T - 1));
            const pt = mapSimToSvg(index, simMinMax.min, 800, 250, { top: 20, right: 20, bottom: 25, left: 45 });
            return (
              <text key={i} className="chart-axis-text" x={pt.x - 5} y="242">t={index + 1}</text>
            );
          })}
          <path className="series-line" d={simSvgPath} style={{ strokeWidth: '1.8' }} />
        </svg>
      </div>

      {/* Resultados de la regresión */}
      {stats && (
        <div key={seriesData.join(',') + '-' + totalSims} style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '1.5rem', alignItems: 'start' }}>
          <div className="glass-card" style={{ margin: 0 }}>
            <h4 style={{ fontFamily: 'var(--heading)', fontWeight: '700', color: 'var(--text-white)', marginBottom: '1rem' }}>
              Resultado de la Estimación OLS (Serie Simulada)
            </h4>
            <div className="sim-stats-grid">
              <div className="stat-card">
                <div className="stat-lbl">{"Pendiente ($\\hat{\\beta}_1$)"}</div>
                <div className="stat-val">{stats.beta1.toFixed(4)}</div>
              </div>
              <div className="stat-card">
                <div className="stat-lbl">Error Est. (SE)</div>
                <div className="stat-val">{stats.se_beta1.toFixed(4)}</div>
              </div>
              <div className="stat-card">
                <div className="stat-lbl">{"Estadístico $\\tau$"}</div>
                <div className={`stat-val ${isRejected ? 'green' : 'orange'}`}>
                  {stats.t_stat.toFixed(4)}
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-lbl">{"Deriva ($\\hat{\\beta}_0$)"}</div>
                <div className="stat-val">{stats.beta0.toFixed(4)}</div>
              </div>
            </div>

            {isRejected ? (
              <div className="decision-card reject" style={{ padding: '1rem' }}>
                <div className="decision-icon" style={{ width: '2rem', height: '2rem' }}>
                  <ShieldIcon isGreen={true} />
                </div>
                <div className="decision-text">
                  <h5 className="decision-title" style={{ fontSize: '0.95rem', marginBottom: '0.15rem' }}>
                    Estacionaria (Se rechaza H0)
                  </h5>
                  <p className="decision-desc" style={{ fontSize: '0.78rem' }}>
                    {`El estadístico $\\tau = ${stats.t_stat.toFixed(3)}$ es menor que el valor crítico $-2.86$. Se descarta la raíz unitaria.`}
                  </p>
                </div>
              </div>
            ) : (
              <div className="decision-card fail-reject" style={{ padding: '1rem' }}>
                <div className="decision-icon" style={{ width: '2rem', height: '2rem' }}>
                  <ShieldIcon isGreen={false} />
                </div>
                <div className="decision-text">
                  <h5 className="decision-title" style={{ fontSize: '0.95rem', marginBottom: '0.15rem' }}>
                    No Estacionaria (No se rechaza H0)
                  </h5>
                  <p className="decision-desc" style={{ fontSize: '0.78rem' }}>
                    {`El estadístico $\\tau = ${stats.t_stat.toFixed(3)}$ no supera la prueba crítica de $-2.86$. Se retiene la raíz unitaria.`}
                  </p>
                </div>
              </div>
            )}
          </div>

          <div className="glass-card" style={{ margin: 0, height: '100%' }}>
            <h4 style={{ fontFamily: 'var(--heading)', fontWeight: '700', color: 'var(--text-white)', marginBottom: '0.75rem' }}>
              Análisis Monte Carlo
            </h4>
            {totalSims > 0 ? (
              <div>
                <p style={{ fontSize: '0.88rem', marginBottom: '1rem' }}>
                  Tras simular <strong>{totalSims}</strong> series independientes con los mismos parámetros:
                </p>
                <div className="grid-3" style={{ gridTemplateColumns: '1fr', gap: '0.75rem', marginBottom: '1rem' }}>
                  <div className="grid-cell" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.8rem' }}>Tasa de Rechazo de H0</span>
                    <span style={{
                      fontWeight: '700',
                      fontSize: '1.25rem',
                      color: params.rho < 1.0 ? 'var(--green)' : 'var(--danger)'
                    }}>
                      {((rejectCount / totalSims) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                  {params.rho < 1.0 ? (
                    <span>
                      {`Esta tasa representa la potencia de la prueba ($1 - \\beta$). Para un proceso estacionario ($\\rho = ${params.rho.toFixed(2)}$), la prueba debería idealmente rechazar $H_0$ el 100% de las veces. Nota cómo disminuye si $\\rho \\to 1$ o si $T$ es muy pequeña (baja potencia).`}
                    </span>
                  ) : (
                    <span>
                      {`Esta tasa representa el error Tipo I de la prueba (falso positivo). Dado que el proceso real posee una raíz unitaria ($\\rho = 1.0$), deberíamos rechazar $H_0$ solo cerca del nivel nominal de significancia (5%).`}
                    </span>
                  )}
                </div>
              </div>
            ) : (
              <div style={{ height: '80%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', textAlign: 'center' }}>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', maxWidth: '200px' }}>
                  {"Haz clic en \"Ejecutar Monte Carlo\" para simular 100 caminos estocásticos y evaluar la potencia estadística de la prueba."}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ============================================================================
   CASE STUDY TAB (VENTA DIARIA TOMATE RIÑON)
   ============================================================================ */
function CaseStudyTab() {
  const [seriesType, setSeriesType] = useState('original'); // 'original' or 'diff'
  const [modelType, setModelType] = useState('hw'); // 'arima' or 'hw'
  const containerRef = useRef(null);

  useEffect(() => {
    if (window.MathJax && window.MathJax.typesetPromise && containerRef.current) {
      window.MathJax.typesetClear([containerRef.current]);
      window.MathJax.typesetPromise([containerRef.current]).catch((err) =>
        console.error('MathJax error during render: ', err)
      );
    }
  }, [seriesType, modelType]);

  // Historical demand (last 21 days of May 2026)
  const historicalData = [
    { date: 'May 11', value: 310.5 },
    { date: 'May 12', value: 1205.2 },
    { date: 'May 13', value: 272.4 },
    { date: 'May 14', value: 312.8 },
    { date: 'May 15', value: 281.3 },
    { date: 'May 16', value: 362.5 },
    { date: 'May 17', value: 382.1 },
    { date: 'May 18', value: 305.8 },
    { date: 'May 19', value: 1195.4 },
    { date: 'May 20', value: 265.9 },
    { date: 'May 21', value: 305.1 },
    { date: 'May 22', value: 275.6 },
    { date: 'May 23', value: 355.0 },
    { date: 'May 24', value: 370.4 },
    { date: 'May 25', value: 300.2 },
    { date: 'May 26', value: 1180.1 },
    { date: 'May 27', value: 260.5 },
    { date: 'May 28', value: 300.9 },
    { date: 'May 29', value: 268.4 },
    { date: 'May 30', value: 352.1 },
    { date: 'May 31', value: 372.4 }
  ];

  // ARIMA projections (15 days of June 2026)
  const arimaProjections = [
    { date: 'Jun 01', value: 485.20 },
    { date: 'Jun 02', value: 472.16 },
    { date: 'Jun 03', value: 475.14 },
    { date: 'Jun 04', value: 474.46 },
    { date: 'Jun 05', value: 474.61 },
    { date: 'Jun 06', value: 474.58 },
    { date: 'Jun 07', value: 474.58 },
    { date: 'Jun 08', value: 474.58 },
    { date: 'Jun 09', value: 474.58 },
    { date: 'Jun 10', value: 474.58 },
    { date: 'Jun 11', value: 474.58 },
    { date: 'Jun 12', value: 474.58 },
    { date: 'Jun 13', value: 474.58 },
    { date: 'Jun 14', value: 474.58 },
    { date: 'Jun 15', value: 474.58 }
  ];

  // Holt-Winters projections (15 days of June 2026)
  const hwProjections = [
    { date: 'Jun 01', value: 308.02 },
    { date: 'Jun 02', value: 1184.95 },
    { date: 'Jun 03', value: 262.45 },
    { date: 'Jun 04', value: 307.04 },
    { date: 'Jun 05', value: 269.77 },
    { date: 'Jun 06', value: 355.96 },
    { date: 'Jun 07', value: 374.21 },
    { date: 'Jun 08', value: 302.61 },
    { date: 'Jun 09', value: 1179.55 },
    { date: 'Jun 10', value: 257.05 },
    { date: 'Jun 11', value: 301.64 },
    { date: 'Jun 12', value: 264.37 },
    { date: 'Jun 13', value: 350.56 },
    { date: 'Jun 14', value: 368.81 },
    { date: 'Jun 15', value: 297.21 }
  ];

  // Calculate differentiated data points
  const diffData = useMemo(() => {
    const list = [];
    for (let i = 1; i < historicalData.length; i++) {
      list.push({
        date: historicalData[i].date,
        value: historicalData[i].value - historicalData[i - 1].value
      });
    }
    return list;
  }, []);

  // Set active chart coordinates
  const activeSeries = useMemo(() => {
    if (seriesType === 'original') {
      const activeProj = modelType === 'arima' ? arimaProjections : hwProjections;
      return [...historicalData, ...activeProj];
    }
    return diffData;
  }, [seriesType, modelType, diffData]);

  // Y bounds calculation
  const minMax = useMemo(() => {
    const vals = activeSeries.map(d => d.value);
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const range = max - min;
    const padding = range * 0.1 || 10;
    return {
      min: min - padding,
      max: max + padding
    };
  }, [activeSeries]);

  const width = 800;
  const height = 250;
  const chartPadding = { top: 20, right: 30, bottom: 30, left: 55 };

  const getSvgCoords = (index, value) => {
    const xRange = activeSeries.length - 1;
    const yRange = minMax.max - minMax.min;
    const x = chartPadding.left + (index / xRange) * (width - chartPadding.left - chartPadding.right);
    const y = chartPadding.top + ((minMax.max - value) / yRange) * (height - chartPadding.top - chartPadding.bottom);
    return { x, y };
  };

  const lineSvgPath = useMemo(() => {
    let list = [];
    if (seriesType === 'original') {
      list = historicalData;
    } else {
      list = diffData;
    }
    const coords = list.map((d, i) => getSvgCoords(i, d.value));
    return coords.map((c, i) => `${i === 0 ? 'M' : 'L'} ${c.x} ${c.y}`).join(' ');
  }, [seriesType, minMax]);

  const forecastSvgPath = useMemo(() => {
    if (seriesType !== 'original') return '';
    const proj = modelType === 'arima' ? arimaProjections : hwProjections;
    const lastHist = historicalData[historicalData.length - 1];
    const firstCoords = getSvgCoords(historicalData.length - 1, lastHist.value);
    const coords = proj.map((d, i) => getSvgCoords(historicalData.length + i, d.value));
    return `M ${firstCoords.x} ${firstCoords.y} ` + coords.map(c => `L ${c.x} ${c.y}`).join(' ');
  }, [seriesType, modelType, minMax]);

  return (
    <div className="tab-panel" ref={containerRef}>
      <h2>Caso de Estudio: Venta Diaria de Tomate Riñón</h2>
      <p style={{ fontFamily: 'var(--serif)', fontStyle: 'italic', marginBottom: '1.5rem' }}>
        Análisis estadístico de una serie de tiempo real con 365 observaciones correspondientes a la demanda diaria de tomate riñón desde el 01/06/2025 hasta el 31/05/2026.
      </p>

      {/* Descriptive Statistics Panel */}
      <div className="glass-card" style={{ marginTop: '0' }}>
        <h3 style={{ fontFamily: 'var(--heading)', fontWeight: '700', color: 'var(--text-white)', marginBottom: '0.75rem' }}>
          Análisis Descriptivo de la Serie de Tiempo Real
        </h3>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-light)', marginBottom: '1.25rem' }}>
          La serie contiene 365 observaciones diarias que abarcan un año completo. Presenta una altísima volatilidad con picos recurrentes y marcados de demanda.
        </p>
        <div className="sim-stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '1rem' }}>
          <div className="stat-card">
            <div className="stat-lbl">Observaciones (T)</div>
            <div className="stat-val">365 días</div>
          </div>
          <div className="stat-card">
            <div className="stat-lbl">Demanda Promedio</div>
            <div className="stat-val">534.91 u</div>
          </div>
          <div className="stat-card">
            <div className="stat-lbl">Desviación Estándar</div>
            <div className="stat-val">378.95 u</div>
          </div>
          <div className="stat-card">
            <div className="stat-lbl">Mínimo / Máximo</div>
            <div className="stat-val" style={{ fontSize: '0.95rem', marginTop: '0.15rem' }}>128.15 / 1922.4</div>
          </div>
          <div className="stat-card">
            <div className="stat-lbl">Coef. Variación</div>
            <div className="stat-val" style={{ color: 'var(--warning)' }}>70.8%</div>
          </div>
        </div>
      </div>

      {/* Toggle Buttons & Visual Chart */}
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
        <div className="nav-tabs" style={{ padding: '0.2rem' }}>
          <button
            className={`tab-btn ${seriesType === 'original' ? 'active' : ''}`}
            onClick={() => setSeriesType('original')}
            style={{ fontSize: '0.8rem', padding: '0.35rem 0.85rem' }}
          >
            Serie Original ($Y_t$)
          </button>
          <button
            className={`tab-btn ${seriesType === 'diff' ? 'active' : ''}`}
            onClick={() => setSeriesType('diff')}
            style={{ fontSize: '0.8rem', padding: '0.35rem 0.85rem' }}
          >
            Serie Diferenciada ($\Delta Y_t$)
          </button>
        </div>

        {seriesType === 'original' && (
          <div className="nav-tabs" style={{ padding: '0.2rem' }}>
            <button
              className={`tab-btn ${modelType === 'hw' ? 'active' : ''}`}
              onClick={() => setModelType('hw')}
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.85rem' }}
            >
              Proyección Holt-Winters (Estacional)
            </button>
            <button
              className={`tab-btn ${modelType === 'arima' ? 'active' : ''}`}
              onClick={() => setModelType('arima')}
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.85rem' }}
            >
              Proyección ARIMA (1,1,1)
            </button>
          </div>
        )}
      </div>

      <div className="chart-container" style={{ marginBottom: '1.5rem' }}>
        <div className="chart-title">
          <span>
            {seriesType === 'original'
              ? `Serie Temporal Real y Proyección a 15 días (${modelType === 'arima' ? 'ARIMA' : 'Holt-Winters'})`
              : 'Primera Diferencia de la Serie Temporal ($\u0394 Y_t$)'}
          </span>
          <div className="chart-legend">
            <div className="legend-item">
              <div className="legend-color" style={{ background: 'var(--accent)' }}></div>
              <span>{seriesType === 'original' ? 'Histórico (Últimos 21 días)' : 'Serie Diferenciada'}</span>
            </div>
            {seriesType === 'original' && (
              <div className="legend-item">
                <div className="legend-color" style={{ background: modelType === 'arima' ? 'var(--danger)' : 'var(--warning)', height: '2px', borderTop: '2px dashed' }}></div>
                <span>Proyección 15 días</span>
              </div>
            )}
          </div>
        </div>

        <svg className="svg-chart" style={{ height: '260px' }} viewBox="0 0 800 260">
          {/* Gridlines Y */}
          {[0, 0.25, 0.5, 0.75, 1].map((p, i) => {
            const yVal = minMax.min + p * (minMax.max - minMax.min);
            const pt = getSvgCoords(0, yVal);
            return (
              <g key={i}>
                <line className="chart-grid-line" x1={chartPadding.left} y1={pt.y} x2={width - chartPadding.right} y2={pt.y} />
                <text className="chart-axis-text" x="5" y={pt.y + 3}>{yVal.toFixed(1)}</text>
              </g>
            );
          })}
          {/* Axis X */}
          <line className="chart-axis-line" x1={chartPadding.left} y1={height - chartPadding.bottom} x2={width - chartPadding.right} y2={height - chartPadding.bottom} />
          {/* Axis X Labels */}
          {activeSeries.map((d, idx) => {
            if (idx % 3 === 0 || idx === activeSeries.length - 1) {
              const pt = getSvgCoords(idx, d.value);
              return (
                <text key={idx} className="chart-axis-text" x={pt.x - 12} y={height - 8}>{d.date}</text>
              );
            }
            return null;
          })}
          {/* Paths */}
          <path className="series-line" d={lineSvgPath} style={{ strokeWidth: '1.8' }} />
          {seriesType === 'original' && (
            <path className="series-line" d={forecastSvgPath} style={{
              stroke: modelType === 'arima' ? 'var(--danger)' : 'var(--warning)',
              strokeWidth: '1.8',
              strokeDasharray: '5 3'
            }} />
          )}
        </svg>
      </div>

      {/* ADF Test Side-by-side results */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>
        {/* original ADF */}
        <div className="glass-card" style={{ margin: 0, borderTop: '3px solid var(--danger)' }}>
          <span className="section-tag" style={{ color: 'var(--danger)' }}>ADF: Serie en Niveles (Original)</span>
          <h4 style={{ fontFamily: 'var(--heading)', fontWeight: '700', color: 'var(--text-white)', marginBottom: '0.75rem' }}>
            Prueba de Raíz Unitaria (ADF) en Niveles
          </h4>
          <div className="math-block" style={{ margin: '0.75rem 0', padding: '0.4rem', fontSize: '0.9rem' }}>
            {"$$y_t = \\rho y_{t-1} + \\epsilon_t$$"}
          </div>
          <p style={{ fontSize: '0.85rem', marginBottom: '0.75rem' }}>
            Resultados del análisis estadístico de la demanda original:
          </p>
          <ul style={{ fontSize: '0.85rem', marginLeft: '1.25rem', marginBottom: '1rem', color: 'var(--text-light)' }}>
            <li>Estadístico de Prueba (t-stat): <strong style={{ color: 'var(--danger)' }}>-1.9764</strong></li>
            <li>Valor p (p-value): <strong style={{ color: 'var(--danger)' }}>0.2970</strong> (mayor a 0.05)</li>
            <li>Rezagos (lags) óptimos: 15 (criterio de AIC)</li>
            <li>Valores críticos (MacKinnon): 1% = -3.4492, 5% = -2.8699, 10% = -2.5712</li>
          </ul>
          <div className="decision-card fail-reject" style={{ padding: '0.75rem', margin: 0 }}>
            <div className="decision-icon" style={{ width: '2rem', height: '2rem' }}>
              <ShieldIcon isGreen={false} />
            </div>
            <div className="decision-text">
              <h5 className="decision-title" style={{ fontSize: '0.9rem', marginBottom: '0.1rem' }}>No Estacionaria (No se rechaza H0)</h5>
              <p className="decision-desc" style={{ fontSize: '0.75rem' }}>
                {"Dado que t-stat (-1.9764) > -2.8699, no podemos rechazar la presencia de raíz unitaria. La serie posee comportamiento no estacionario."}
              </p>
            </div>
          </div>
        </div>

        {/* Diff ADF */}
        <div className="glass-card" style={{ margin: 0, borderTop: '3px solid var(--green)' }}>
          <span className="section-tag" style={{ color: 'var(--green)' }}>ADF: Serie Diferenciada</span>
          <h4 style={{ fontFamily: 'var(--heading)', fontWeight: '700', color: 'var(--text-white)', marginBottom: '0.75rem' }}>
            Prueba de Raíz Unitaria (ADF) en Diferencias
          </h4>
          <div className="math-block" style={{ margin: '0.75rem 0', padding: '0.4rem', fontSize: '0.9rem' }}>
            {"$$\\Delta y_t = \\phi y_{t-1} + \\epsilon_t$$"}
          </div>
          <p style={{ fontSize: '0.85rem', marginBottom: '0.75rem' }}>
            {"Resultados del análisis estadístico tras aplicar una diferencia ($\\Delta Y_t = Y_t - Y_{t-1}$):"}
          </p>
          <ul style={{ fontSize: '0.85rem', marginLeft: '1.25rem', marginBottom: '1rem', color: 'var(--text-light)' }}>
            <li>Estadístico de Prueba (t-stat): <strong style={{ color: 'var(--green)' }}>-7.3553</strong></li>
            <li>Valor p (p-value): <strong style={{ color: 'var(--green)' }}>0.0000</strong> (altamente significativo)</li>
            <li>Rezagos (lags) óptimos: 14 (criterio de AIC)</li>
            <li>Valores críticos (MacKinnon): 1% = -3.4492, 5% = -2.8699, 10% = -2.5712</li>
          </ul>
          <div className="decision-card reject" style={{ padding: '0.75rem', margin: 0 }}>
            <div className="decision-icon" style={{ width: '2rem', height: '2rem' }}>
              <ShieldIcon isGreen={true} />
            </div>
            <div className="decision-text">
              <h5 className="decision-title" style={{ fontSize: '0.9rem', marginBottom: '0.1rem' }}>Serie Estacionaria (Se rechaza H0)</h5>
              <p className="decision-desc" style={{ fontSize: '0.75rem' }}>
                {"El estadístico de prueba -7.3553 es significativamente menor que -3.4492. Se rechaza la raíz unitaria. La serie diferenciada es estacionaria (I(0))."}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Forecast comparison section */}
      <div className="glass-card">
        <h3 style={{ fontFamily: 'var(--heading)', fontWeight: '700', color: 'var(--text-white)', marginBottom: '1rem' }}>
          Comparación de Modelos de Proyección: ARIMA(1,1,1) vs. Holt-Winters
        </h3>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-light)', marginBottom: '1.25rem', lineHeight: '1.6' }}>
          La demanda diaria del Tomate Riñón presenta un fuerte componente estacional de carácter semanal.
          Al proyectar 15 días, observamos un fenómeno operativo crítico: el modelo <strong>ARIMA(1,1,1)</strong> se aplana casi por completo a partir del segundo día
          (estancándose en $\approx 474$ unidades), lo que limita su utilidad práctica.
          <br /><br />
          Para solucionarlo, se implementó el modelo de suavización exponencial de <strong>Holt-Winters</strong> estacional aditivo con periodos estacionales de 7 días, logrando replicar con éxito los picos de venta de los días martes y valles de media semana.
        </p>

        <div className="premium-table-container">
          <table className="premium-table">
            <thead>
              <tr>
                <th>Fecha</th>
                <th>Día de la Semana</th>
                <th>Proyección ARIMA (1,1,1)</th>
                <th>Proyección Holt-Winters (Estacional)</th>
                <th>Comparativa de Comportamiento</th>
              </tr>
            </thead>
            <tbody>
              {arimaProjections.map((item, idx) => {
                const hwVal = hwProjections[idx].value;
                const isPeak = hwVal > 1000;
                const isValley = hwVal < 300;
                let dayName = 'Lunes';
                switch (idx % 7) {
                  case 0: dayName = 'Lunes'; break;
                  case 1: dayName = 'Martes'; break;
                  case 2: dayName = 'Miércoles'; break;
                  case 3: dayName = 'Jueves'; break;
                  case 4: dayName = 'Viernes'; break;
                  case 5: dayName = 'Sábado'; break;
                  case 6: dayName = 'Domingo'; break;
                }

                return (
                  <tr key={idx}>
                    <td>{item.date}</td>
                    <td style={{ fontWeight: isPeak ? '600' : '400', color: isPeak ? 'var(--warning)' : 'var(--text-light)' }}>
                      {dayName}
                    </td>
                    <td>{item.value.toFixed(2)} u</td>
                    <td style={{ fontWeight: isPeak || isValley ? '700' : '400', color: isPeak ? 'var(--warning)' : (isValley ? 'var(--accent)' : 'var(--text-light)') }}>
                      {hwVal.toFixed(2)} u
                    </td>
                    <td style={{ fontSize: '0.8rem', textAlign: 'left', paddingLeft: '1.5rem', color: isPeak ? 'var(--warning)' : (isValley ? 'var(--accent)' : 'var(--text-muted)') }}>
                      {isPeak ? '★ Pico de demanda semanal (Martes)' : (isValley ? '▼ Valle de demanda (Miércoles)' : 'Demanda regular')}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default App;
