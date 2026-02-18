/* ============================================
   MECHANO-VELOCITY DASHBOARD — MAIN JS
   Charts, KaTeX equations, scroll animations
   ============================================ */

// ---- Scroll Animations (Intersection Observer) ----
function initScrollAnimations() {
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        },
        { threshold: 0.1, rootMargin: '0px 0px -50px 0px' }
    );

    document.querySelectorAll('.animate-in').forEach((el) => observer.observe(el));
}

// ---- Navbar Scroll Effect ----
function initNavbar() {
    const navbar = document.getElementById('navbar');
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        // Scrolled state
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        // Active section highlighting
        let current = '';
        sections.forEach((section) => {
            const sectionTop = section.offsetTop - 150;
            if (window.scrollY >= sectionTop) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach((link) => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

// ---- KaTeX Equation Rendering ----
function initEquations() {
    if (typeof katex === 'undefined') {
        console.warn('KaTeX not loaded, retrying...');
        setTimeout(initEquations, 500);
        return;
    }

    const equations = {
        'eq-density': 'D_i = (\\alpha \\cdot \\text{COL1A1} + \\alpha \\cdot \\text{COL1A2}) \\times (1 + \\beta \\cdot \\text{LOX}) - (\\gamma \\cdot \\text{MMP9})',
        'eq-sigmoid': 'R_i = \\frac{1}{1 + e^{-(D_i - \\mu)}}'
    };

    for (const [id, tex] of Object.entries(equations)) {
        const el = document.getElementById(id);
        if (el) {
            katex.render(tex, el, {
                displayMode: true,
                throwOnError: false,
                output: 'html',
            });
        }
    }
}

// ---- Chart.js Configuration ----
const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            display: false,
        },
    },
    scales: {
        x: {
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: { color: 'rgba(240,240,248,0.5)', font: { size: 11, family: 'Inter' } },
        },
        y: {
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: { color: 'rgba(240,240,248,0.5)', font: { size: 11, family: 'Inter' } },
        },
    },
};

// ---- QC Metrics Chart ----
function initQCChart() {
    const ctx = document.getElementById('qcChart');
    if (!ctx) return;

    // Simulated QC distribution data (realistic for Visium breast cancer)
    const bins = Array.from({ length: 25 }, (_, i) => (i * 400).toString());
    const counts = [
        12, 18, 45, 85, 140, 210, 320, 450, 520, 480,
        420, 350, 290, 220, 170, 120, 85, 60, 42, 30,
        20, 14, 8, 5, 3
    ];

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: bins,
            datasets: [{
                label: 'Spots',
                data: counts,
                backgroundColor: counts.map((_, i) => {
                    const val = i * 400;
                    if (val < 500) return 'rgba(239, 68, 68, 0.6)'; // below threshold
                    return 'rgba(0, 212, 255, 0.4)';
                }),
                borderColor: counts.map((_, i) => {
                    const val = i * 400;
                    if (val < 500) return 'rgba(239, 68, 68, 0.8)';
                    return 'rgba(0, 212, 255, 0.6)';
                }),
                borderWidth: 1,
                borderRadius: 3,
            }],
        },
        options: {
            ...chartDefaults,
            plugins: {
                ...chartDefaults.plugins,
                tooltip: {
                    backgroundColor: 'rgba(12,12,29,0.95)',
                    titleFont: { family: 'Inter' },
                    bodyFont: { family: 'Inter' },
                    borderColor: 'rgba(0,212,255,0.2)',
                    borderWidth: 1,
                    callbacks: {
                        title: (items) => `UMI Count: ${items[0].label}`,
                        label: (item) => `${item.raw} spots`,
                    }
                },
                annotation: undefined,
            },
            scales: {
                x: {
                    ...chartDefaults.scales.x,
                    title: {
                        display: true,
                        text: 'Total UMI Counts',
                        color: 'rgba(240,240,248,0.5)',
                        font: { size: 11, family: 'Inter' }
                    },
                    ticks: {
                        ...chartDefaults.scales.x.ticks,
                        maxTicksLimit: 8,
                    },
                },
                y: {
                    ...chartDefaults.scales.y,
                    title: {
                        display: true,
                        text: 'Number of Spots',
                        color: 'rgba(240,240,248,0.5)',
                        font: { size: 11, family: 'Inter' }
                    },
                },
            },
        },
    });
}

// ---- Resistance Distribution Chart ----
function initResistanceChart() {
    const ctx = document.getElementById('resistanceChart');
    if (!ctx) return;

    // Simulated sigmoid-shaped resistance distribution
    const labels = [];
    const data = [];
    for (let i = 0; i <= 20; i++) {
        const r = i / 20;
        labels.push(r.toFixed(2));
        // Bimodal distribution: many low, some high
        const low = 300 * Math.exp(-((r - 0.25) ** 2) / 0.03);
        const high = 150 * Math.exp(-((r - 0.75) ** 2) / 0.04);
        const mid = 80 * Math.exp(-((r - 0.5) ** 2) / 0.02);
        data.push(Math.round(low + mid + high));
    }

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Spots',
                data,
                backgroundColor: data.map((_, i) => {
                    const r = i / 20;
                    if (r < 0.35) return 'rgba(59, 130, 246, 0.5)';   // Fluid
                    if (r > 0.65) return 'rgba(239, 68, 68, 0.5)';    // Wall
                    return 'rgba(107, 114, 128, 0.4)';                  // Normal
                }),
                borderColor: data.map((_, i) => {
                    const r = i / 20;
                    if (r < 0.35) return 'rgba(59, 130, 246, 0.7)';
                    if (r > 0.65) return 'rgba(239, 68, 68, 0.7)';
                    return 'rgba(107, 114, 128, 0.6)';
                }),
                borderWidth: 1,
                borderRadius: 3,
            }],
        },
        options: {
            ...chartDefaults,
            plugins: {
                ...chartDefaults.plugins,
                tooltip: {
                    backgroundColor: 'rgba(12,12,29,0.95)',
                    titleFont: { family: 'Inter' },
                    bodyFont: { family: 'Inter' },
                    borderColor: 'rgba(0,212,255,0.2)',
                    borderWidth: 1,
                    callbacks: {
                        title: (items) => `Resistance: ${items[0].label}`,
                        label: (item) => {
                            const r = parseFloat(item.label);
                            let cat = 'Normal';
                            if (r < 0.35) cat = '🔵 Fluid';
                            else if (r > 0.65) cat = '🔴 Wall';
                            return `${item.raw} spots (${cat})`;
                        },
                    }
                },
            },
            scales: {
                x: {
                    ...chartDefaults.scales.x,
                    title: {
                        display: true,
                        text: 'Resistance Score (R)',
                        color: 'rgba(240,240,248,0.5)',
                        font: { size: 11, family: 'Inter' }
                    },
                    ticks: {
                        ...chartDefaults.scales.x.ticks,
                        maxTicksLimit: 6,
                    },
                },
                y: {
                    ...chartDefaults.scales.y,
                    title: {
                        display: true,
                        text: 'Number of Spots',
                        color: 'rgba(240,240,248,0.5)',
                        font: { size: 11, family: 'Inter' }
                    },
                },
            },
        },
    });
}

// ---- Drug Simulation Chart ----
function initDrugChart() {
    const ctx = document.getElementById('drugChart');
    if (!ctx) return;

    const categories = ['Fluid\n(R < 0.35)', 'Normal\n(0.35-0.65)', 'Wall\n(R > 0.65)'];
    const original = [1420, 1580, 800];
    const simulated = [1750, 1680, 370];

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: categories,
            datasets: [
                {
                    label: 'Before (Original)',
                    data: original,
                    backgroundColor: 'rgba(139, 92, 246, 0.5)',
                    borderColor: 'rgba(139, 92, 246, 0.7)',
                    borderWidth: 1,
                    borderRadius: 6,
                },
                {
                    label: 'After LOX Inhibitor',
                    data: simulated,
                    backgroundColor: 'rgba(16, 185, 129, 0.5)',
                    borderColor: 'rgba(16, 185, 129, 0.7)',
                    borderWidth: 1,
                    borderRadius: 6,
                },
            ],
        },
        options: {
            ...chartDefaults,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: 'rgba(240,240,248,0.6)',
                        font: { size: 11, family: 'Inter' },
                        boxWidth: 12,
                        boxHeight: 12,
                        borderRadius: 3,
                        useBorderRadius: true,
                        padding: 16,
                    },
                },
                tooltip: {
                    backgroundColor: 'rgba(12,12,29,0.95)',
                    titleFont: { family: 'Inter' },
                    bodyFont: { family: 'Inter' },
                    borderColor: 'rgba(0,212,255,0.2)',
                    borderWidth: 1,
                },
            },
            scales: {
                x: {
                    ...chartDefaults.scales.x,
                    ticks: {
                        ...chartDefaults.scales.x.ticks,
                        maxRotation: 0,
                    },
                },
                y: {
                    ...chartDefaults.scales.y,
                    title: {
                        display: true,
                        text: 'Number of Spots',
                        color: 'rgba(240,240,248,0.5)',
                        font: { size: 11, family: 'Inter' }
                    },
                },
            },
        },
    });
}

// ---- Initialize Everything ----
document.addEventListener('DOMContentLoaded', () => {
    initScrollAnimations();
    initNavbar();

    // Wait for external scripts to load
    const initCharts = () => {
        if (typeof Chart !== 'undefined') {
            // Set Chart.js global defaults
            Chart.defaults.font.family = 'Inter';
            Chart.defaults.color = 'rgba(240,240,248,0.5)';

            initQCChart();
            initResistanceChart();
            initDrugChart();
        } else {
            setTimeout(initCharts, 300);
        }
    };

    initCharts();
    initEquations();

    // Trigger hero animation immediately
    setTimeout(() => {
        document.querySelector('.hero-content')?.classList.add('visible');
    }, 200);
});
