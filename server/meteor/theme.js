(function () {
    'use strict';

    function initTheme() {
        const body = document.body;
        const canvas = document.getElementById('starfield');
        const themeRadios = document.querySelectorAll('input[name="nmn-theme"]');
        const themes = {
            classic: { stars: false, starColor: '#ffffff', shootingStarColor: '#ffffff', bg: null },
            night:   { stars: true,  starColor: '#ffffff', shootingStarColor: '#ffd166', radiant: { x: 0.2, y: 0.2 }, bg: { type: 'radial', stops: [[0, '#1b2735'], [1, '#090a0f']] } }
        };

        let starfield = null;
        if (canvas) starfield = initStarfield(canvas);

        function applyTheme(name) {
            const theme = themes[name] || themes.classic;
            body.classList.forEach(cls => { if (cls.startsWith('theme-')) body.classList.remove(cls); });
            body.classList.remove('theme-dark');
            body.classList.add(`theme-${name}`);
            if (name !== 'classic') body.classList.add('theme-dark');
            localStorage.setItem('nmn-meteor-theme', name);
            themeRadios.forEach(radio => {
                const checked = radio.value === name;
                radio.checked = checked;
                const label = radio.closest('label');
                if (label) label.classList.toggle('active', checked);
            });
            if (starfield) starfield.setOptions(theme);
            // map.jpg + spd_acc.jpg: pixel-transform instead of CSS invert
            // — white bg goes transparent, text/grid goes light, colours
            // keep their hue (see themeMapImage).
            const dark = name !== 'classic';
            document.querySelectorAll('img.plot').forEach(img => {
                if (/(map|spd_acc|height|posvstime|wind_profile)\.jpg/.test(img.dataset.origSrc || img.src))
                    themeMapImage(img, dark);
            });
            // Swap media sources that declare a night variant (data-night-src).
            document.querySelectorAll('source[data-night-src]').forEach(src => {
                if (src.dataset.daySrc === undefined) src.dataset.daySrc = src.getAttribute('src');
                const target = name === 'classic' ? src.dataset.daySrc : src.dataset.nightSrc;
                if (target && src.getAttribute('src') !== target) {
                    src.setAttribute('src', target);
                    const video = src.closest('video');
                    if (video) { video.load(); video.play().catch(() => {}); }
                }
            });
        }

        themeRadios.forEach(radio => {
            radio.addEventListener('change', () => { if (radio.checked) applyTheme(radio.value); });
        });

        const param = new URLSearchParams(location.search).get('theme');
        const saved = localStorage.getItem('nmn-meteor-theme') || 'classic';
        applyTheme(param && themes[param] ? param : saved);
    }

    // Dark-theme transform for the static map image: achromatic pixels
    // (white background, black text/grid) become white with alpha = 1-lum,
    // so white turns transparent (navy card shows through) and black turns
    // opaque white. Chromatic terrain colours are kept unchanged.
    function themeMapImage(img, dark) {
        if (!img.dataset.origSrc) img.dataset.origSrc = img.getAttribute('src');
        const isMap = /(?:^|[/_])map\.jpg(?:[?#]|$)/.test(img.dataset.origSrc);
        img.classList.toggle('nmn-darkmap', dark);
        if (!dark) {
            img.style.opacity = '';
            if (img.src !== img.dataset.origSrc) img.src = img.dataset.origSrc;
            return;
        }
        img.style.opacity = '0';  // hidden until the night version is swapped in
        const convert = () => {
            const c = document.createElement('canvas');
            c.width = img.naturalWidth;
            c.height = img.naturalHeight;
            const ctx = c.getContext('2d');
            ctx.drawImage(img, 0, 0);
            const d = ctx.getImageData(0, 0, c.width, c.height);
            const px = d.data;
            const W = c.width, H = c.height;
            // pass 1: chroma mask + integral image for a 7x7 box count,
            // so isolated coloured specks can be demoted to achromatic
            const mask = new Uint8Array(W * H);
            for (let i = 0, p = 0; p < px.length; p += 4, i++) {
                const mx = Math.max(px[p], px[p + 1], px[p + 2]);
                const mn = Math.min(px[p], px[p + 1], px[p + 2]);
                mask[i] = isMap
                    ? !((mx - mn < 55) || (mx - mn < 80 && mx < 210))
                    : mx - mn > 8;
            }
            const integ = new Int32Array((W + 1) * (H + 1));
            for (let y = 0; y < H; y++)
                for (let x = 0; x < W; x++)
                    integ[(y + 1) * (W + 1) + x + 1] =
                        integ[y * (W + 1) + x + 1] + integ[(y + 1) * (W + 1) + x]
                        - integ[y * (W + 1) + x] + mask[y * W + x];
            const chromNeighbours = (x, y) => {
                const x0 = Math.max(0, x - 3), y0 = Math.max(0, y - 3);
                const x1 = Math.min(W - 1, x + 3), y1 = Math.min(H - 1, y + 3);
                return integ[(y1 + 1) * (W + 1) + x1 + 1]
                     - integ[y0 * (W + 1) + x1 + 1]
                     - integ[(y1 + 1) * (W + 1) + x0]
                     + integ[y0 * (W + 1) + x0];
            };
            // For the map, mark pixels outside the map frame: near-white
            // areas connected to the image border (figure margins, outside
            // the axes). Interior light areas — sea, uncovered tiles —
            // keep the opaque dimmed blend like the map.html surface.
            let outside = null;
            if (isMap) {
                outside = new Uint8Array(W * H);
                const isLight = (i) => { const p = i * 4;
                    return px[p] > 225 && px[p + 1] > 225 && px[p + 2] > 225; };
                const queue = new Int32Array(W * H);
                let qh = 0, qt = 0;
                const seed = (i) => { if (!outside[i] && isLight(i)) { outside[i] = 1; queue[qt++] = i; } };
                for (let x = 0; x < W; x++) { seed(x); seed((H - 1) * W + x); }
                for (let y = 0; y < H; y++) { seed(y * W); seed(y * W + W - 1); }
                while (qh < qt) {
                    const i = queue[qh++], x = i % W, y = (i / W) | 0;
                    if (x > 0)     { const n = i - 1; if (!outside[n] && isLight(n)) { outside[n] = 1; queue[qt++] = n; } }
                    if (x < W - 1) { const n = i + 1; if (!outside[n] && isLight(n)) { outside[n] = 1; queue[qt++] = n; } }
                    if (y > 0)     { const n = i - W; if (!outside[n] && isLight(n)) { outside[n] = 1; queue[qt++] = n; } }
                    if (y < H - 1) { const n = i + W; if (!outside[n] && isLight(n)) { outside[n] = 1; queue[qt++] = n; } }
                }
            }
            // pass 2: transform
            for (let i = 0, p = 0; p < px.length; p += 4, i++) {
                const r = px[p], g = px[p + 1], b = px[p + 2];
                const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
                let achrom = !mask[i];
                if (isMap && !achrom) {
                    // few chromatic neighbours -> speck, not a feature
                    const frac = chromNeighbours(i % W, (i / W) | 0)
                        / ((Math.min(W - 1, (i % W) + 3) - Math.max(0, (i % W) - 3) + 1)
                         * (Math.min(H - 1, ((i / W) | 0) + 3) - Math.max(0, ((i / W) | 0) - 3) + 1));
                    if (frac < 0.22) achrom = true;
                }
                if (achrom && isMap && !(outside && outside[i])) {
                    // Interior achromatic pixels: ink-coverage ramp from the
                    // dimmed backdrop (light bg / water / tile edges) toward
                    // opaque near-white text. A smooth ramp keeps the map's
                    // anti-aliasing intact instead of a hard brightness cut.
                    const cov = Math.min(1, Math.max(0, (200 - mx) / 160));
                    const bgr = r * 0.55 + 20 * 0.45;
                    const bgg = g * 0.55 + 30 * 0.45;
                    const bgb = b * 0.55 + 40 * 0.45;
                    px[p]     = Math.round(bgr + (235 - bgr) * cov);
                    px[p + 1] = Math.round(bgg + (238 - bgg) * cov);
                    px[p + 2] = Math.round(bgb + (244 - bgb) * cov);
                } else if (achrom) {
                    // white fades out, dark text/borders go bright; tint
                    // follows the hue hint so water stays slightly blue
                    // and land slightly green
                    if (b > r && b > g)      { px[p] = 190; px[p+1] = 215; px[p+2] = 245; }
                    else if (g > r && g > b) { px[p] = 205; px[p+1] = 240; px[p+2] = 210; }
                    else                     { px[p] = 225; px[p+1] = 228; px[p+2] = 232; }
                    px[p + 3] = Math.round(255 * Math.pow(1 - mx / 255, 0.55));
                } else {
                    // chromatic: normalise brightness toward 200 keeping hue
                    // (dark sight lines brighten, bright overlays dim)
                    let sc = Math.min(1.7, Math.max(0.55, 200 / Math.max(mx, 1)));
                    if (isMap && (mx - mn) < 90) {
                        // Muted terrain colours: match the interactive
                        // map.html night look where the surface is dimmed
                        // to 55% opacity over the #141e28 backdrop.
                        px[p]     = Math.round(r * 0.55 + 20 * 0.45);
                        px[p + 1] = Math.round(g * 0.55 + 30 * 0.45);
                        px[p + 2] = Math.round(b * 0.55 + 40 * 0.45);
                        continue;
                        // saturated lines/markers fall through: keep them
                        // vivid like the Plotly traces in the iframe
                    }
                    if (!isMap) {
                        // large, low-saturation light fills: blend into page
                        if (sc < 0.9 && mx > 150 && (mx - mn) < 160) sc *= 0.35;
                        // saturated blue lines have poor contrast on navy:
                        // remap to gold, keeping per-pixel brightness
                        else if (mx === b && b - r > 70 && b - g > 70) {
                            const t = Math.min(255, b * sc);
                            px[p]     = Math.round(t * 0.92);
                            px[p + 1] = Math.round(t * 0.76);
                            px[p + 2] = Math.round(t * 0.22);
                            continue;
                        }
                    }
                    px[p]     = Math.round(Math.min(255, r * sc));
                    px[p + 1] = Math.round(Math.min(255, g * sc));
                    px[p + 2] = Math.round(Math.min(255, b * sc));
                }
            }
            ctx.putImageData(d, 0, 0);
            img.src = c.toDataURL('image/png');
            img.style.opacity = '';
        };
        if (img.complete && img.naturalWidth) convert();
        else img.addEventListener('load', convert, { once: true });
    }

    function initStarfield(canvas) {
        const ctx = canvas.getContext('2d');
        let width = 0, height = 0;
        let stars = [];
        let shootingStars = [];
        let opts = { stars: false, starColor: '#ffffff', shootingStarColor: '#ffffff', bg: null };
        let rafId = null;

        const hexToRgb = (hex) => {
            const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
            return m ? `${parseInt(m[1], 16)}, ${parseInt(m[2], 16)}, ${parseInt(m[3], 16)}` : '255, 255, 255';
        };

        function resize() {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        }

        function createStars() {
            stars = [];
            const count = Math.floor((width * height) / 3500);
            for (let i = 0; i < count; i++) {
                // Bias toward faint stars: most are small and dim.
                const brightness = Math.pow(Math.random(), 2);
                stars.push({
                    x: Math.random() * width,
                    y: Math.random() * height,
                    r: 0.2 + brightness * 1.4,
                    baseAlpha: 0.2 + brightness * 0.6,
                    phase: Math.random() * Math.PI * 2,
                    speed: Math.random() * 0.02 + 0.005,
                    state: 'stable',
                    stateTimer: Math.floor(Math.random() * 240) + 60
                });
            }
        }

        function spawnShootingStar() {
            if (!opts.stars) return;
            let x, y, angle;
            if (opts.radiant) {
                const rx = width * opts.radiant.x;
                const ry = height * opts.radiant.y;
                // Start somewhere along a random ray from the radiant point.
                angle = Math.random() * Math.PI * 2;
                const distance = Math.random() * Math.max(width, height) * 0.8;
                x = rx + Math.cos(angle) * distance;
                y = ry + Math.sin(angle) * distance;
            } else {
                x = Math.random() * width;
                y = Math.random() * height * 0.6;
                angle = Math.PI / 4 + Math.random() * Math.PI / 6;
            }
            shootingStars.push({
                x: x,
                y: y,
                len: Math.random() * 80 + 40,
                speed: Math.random() * 12 + 8,
                angle: angle,
                life: 1
            });
        }

        function draw() {
            if (opts.bg) {
                let grd;
                if (opts.bg.type === 'radial') {
                    grd = ctx.createRadialGradient(width / 2, height, 0, width / 2, height / 2, Math.max(width, height));
                } else {
                    grd = ctx.createLinearGradient(0, 0, 0, height);
                }
                opts.bg.stops.forEach(([pos, color]) => grd.addColorStop(pos, color));
                ctx.fillStyle = grd;
                ctx.fillRect(0, 0, width, height);
            } else {
                ctx.clearRect(0, 0, width, height);
            }
            if (opts.stars) {
                const [r, g, b] = hexToRgb(opts.starColor).split(',').map(s => parseInt(s.trim(), 10));
                stars.forEach(star => {
                    star.stateTimer--;
                    if (star.stateTimer <= 0) {
                        if (star.state === 'stable') {
                            star.state = 'flicker';
                            star.stateTimer = Math.floor(Math.random() * 30) + 15;
                        } else {
                            star.state = 'stable';
                            star.stateTimer = Math.floor(Math.random() * 240) + 60;
                        }
                    }
                    star.phase += star.speed * (star.state === 'flicker' ? 6 : 1);
                    let flicker = 0;
                    if (star.state === 'flicker') {
                        flicker = (Math.random() - 0.5) * 0.35;
                    }
                    const slowPulse = Math.sin(star.phase) * 0.04;
                    let alpha = star.baseAlpha + slowPulse + flicker;
                    if (alpha < 0) alpha = 0;
                    if (alpha > 1) alpha = 1;
                    ctx.beginPath();
                    ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
                    ctx.fill();
                });

                const [sr, sg, sb] = hexToRgb(opts.shootingStarColor).split(',').map(s => parseInt(s.trim(), 10));
                for (let i = shootingStars.length - 1; i >= 0; i--) {
                    const s = shootingStars[i];
                    s.x += Math.cos(s.angle) * s.speed;
                    s.y += Math.sin(s.angle) * s.speed;
                    s.life -= 0.02;
                    const tailX = s.x - Math.cos(s.angle) * s.len;
                    const tailY = s.y - Math.sin(s.angle) * s.len;
                    const grad = ctx.createLinearGradient(s.x, s.y, tailX, tailY);
                    grad.addColorStop(0, `rgba(${sr}, ${sg}, ${sb}, ${Math.max(0, s.life)})`);
                    grad.addColorStop(1, `rgba(${sr}, ${sg}, ${sb}, 0)`);
                    ctx.strokeStyle = grad;
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(s.x, s.y);
                    ctx.lineTo(tailX, tailY);
                    ctx.stroke();
                    if (s.life <= 0 || s.x > width + s.len || s.y > height + s.len) {
                        shootingStars.splice(i, 1);
                    }
                }
                if (Math.random() < 0.02) spawnShootingStar();
            }
            rafId = requestAnimationFrame(draw);
        }

        function setOptions(options) {
            opts = options || opts;
            if (opts.stars) {
                resize();
                createStars();
            } else {
                shootingStars = [];
            }
        }

        window.addEventListener('resize', () => { resize(); createStars(); });
        resize();
        createStars();
        draw();

        return { setOptions };
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initTheme);
    } else {
        initTheme();
    }
})();
