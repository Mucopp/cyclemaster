/**
 * ═══════════════════════════════════════════════════════════════════════════
 * CycleMaster Deep Learning Engine  —  ml.js
 * TensorFlow.js tabanlı, tarayıcı içi sinir ağı tahmin motoru
 *
 * Mimari: Dense(64, relu) → Dropout(0.25) → Dense(32, relu) → Dropout(0.20) →
 *         Dense(16, relu) → Dense(1, sigmoid)
 *
 * Özellikler (14): RSI, MACD, BB%B, EMA oranı, hacim oranı, ATR oranı,
 *                  CMF, OBV yönü, mum gövdesi, wick oranı, StochRSI,
 *                  1/3/5-bar getiri, momentum(5)
 *
 * Eğitim: Son 150 bar, 45 epoch, 30 dk'da bir yeniden eğitim
 *
 * Kalıcılık (Persistence): Model mimarisi ve ağırlıkları tarayıcı IndexedDB'sine
 *                          (indexeddb://cyclemaster-ml-model) kaydedilir.
 *                          Doğruluk ve eğitim zamanı localStorage'da tutulur.
 *                          Böylece sayfa yenilense veya GitHub Pages'e yüklense bile
 *                          hafızadaki model anında yüklenir ve tahmin üretir.
 * ═══════════════════════════════════════════════════════════════════════════
 */

/* global tf, calcRSI, calcEMA, calcBB, calcATR, calcCMF, calcOBV,
          calcMOM, calcStochRSI, calcSMA */

// ── State ──────────────────────────────────────────────────────────────────
let _mlModel    = null;   // Eğitilmiş veya yüklenmiş TF.js modeli
let _mlReady    = false;  // Model hazır mı? (bellekte veya diskte var mı)
let _mlScore    = 50;     // Son tahmin (0-100, >50 boğa)
let _mlDir      = 'NÖTR'; // Son tahmin yönü
let _mlTrainTs  = 0;      // Son eğitim timestamp'i (ms)
let _mlAccuracy = null;   // Son eğitim doğruluğu
let _mlTraining = false;  // Şu an eğitiliyor mu? (çift eğitim önleme)

// ── UI Yardımcıları ────────────────────────────────────────────────────────
function _mlSetBadge(text, cls, dotActive) {
    const badge = document.getElementById('ml-badge');
    const dot   = document.getElementById('ml-dot');
    if (badge) { badge.textContent = text; badge.className = 'ind-badge ' + cls; }
    if (dot)   { dot.className = 'dot' + (dotActive ? ' active' : ''); }
}

function _mlSetStatus(text) {
    const el = document.getElementById('ml-status');
    if (el) el.textContent = text;
}

// ── Özellik Vektörü (14 boyutlu, tanh ile sınırlandırılmış) ───────────────
function _mlFeatures(closes, highs, lows, volumes, i) {
    if (i < 30) return null;
    const c = closes, h = highs, l = lows, v = volumes;

    // Getiriler
    const ret1 = (c[i] - c[i-1]) / (c[i-1] || 1);
    const ret3 = (c[i] - c[i-3]) / (c[i-3] || 1);
    const ret5 = (c[i] - c[i-5]) / (c[i-5] || 1);

    // RSI(14) → 0-1
    const rsiArr  = calcRSI(c.slice(0, i+1), 14);
    const rsi     = (rsiArr[rsiArr.length - 1] || 50) / 100;

    // StochRSI → 0-1
    const stochArr = calcStochRSI(rsiArr, 14);
    const stoch    = (stochArr[stochArr.length - 1] || 50) / 100;

    // MACD histogram (fiyata normalize)
    const e12     = calcEMA(c.slice(0, i+1), 12);
    const e26     = calcEMA(c.slice(0, i+1), 26);
    const macdRaw = e12[e12.length-1] - e26[e26.length-1];
    const macd    = macdRaw / (c[i] || 1);

    // Bollinger %B → 0-1
    const bb      = calcBB(c.slice(0, i+1), 20, 2);
    const bbRng   = (bb.upper - bb.lower) || 1;
    const bbPct   = Math.min(1, Math.max(0, (c[i] - bb.lower) / bbRng));

    // EMA5/13 fark oranı
    const ema5v   = calcEMA(c.slice(0, i+1), 5);
    const ema13v  = calcEMA(c.slice(0, i+1), 13);
    const emaRatio = (ema5v[ema5v.length-1] - ema13v[ema13v.length-1]) / (c[i] || 1);

    // Hacim oranı (son bar / son 10 bar ort.)
    const vol10   = v.slice(Math.max(0, i-10), i).reduce((a,b)=>a+b,0) / 10 || 1;
    const volRatio = v[i] / vol10;

    // ATR oranı (son bar / fiyat)
    const atrArr  = calcATR(h.slice(0, i+1), l.slice(0, i+1), c.slice(0, i+1), 14);
    const atrRatio = (atrArr[atrArr.length-1] || 1) / (c[i] || 1);

    // CMF
    const cmfArr  = calcCMF(h.slice(0, i+1), l.slice(0, i+1), c.slice(0, i+1), v.slice(0, i+1), 14);
    const cmf     = cmfArr.length > 0 ? cmfArr[cmfArr.length-1] : 0;

    // OBV yönü (son 5 bar)
    const obvSl   = calcOBV(c.slice(i-5, i+1), v.slice(i-5, i+1));
    const obvDir  = (obvSl[obvSl.length-1] - obvSl[0]) / (Math.abs(obvSl[0]) || 1);

    // Mum gövdesi yönü ve büyüklüğü
    const body    = (c[i] - c[i-1]) / (c[i-1] || 1);

    // Wick oranı (volatilite)
    const wickR   = (h[i] - l[i]) / (c[i] || 1);

    return [
        Math.tanh(ret1  * 20),           // 1-bar getiri
        Math.tanh(ret3  * 10),           // 3-bar getiri
        Math.tanh(ret5  * 7),            // 5-bar getiri
        rsi,                             // RSI 0-1
        stoch,                           // StochRSI 0-1
        Math.tanh(macd  * 100),          // MACD normalize
        bbPct,                           // BB %B 0-1
        Math.tanh(emaRatio * 100),       // EMA oranı
        Math.tanh(volRatio - 1),         // Hacim oranı
        Math.tanh(atrRatio * 50),        // ATR oranı
        Math.tanh(cmf  * 3),             // CMF
        Math.tanh(obvDir),               // OBV yönü
        Math.tanh(body * 30),            // Mum gövdesi
        Math.tanh(wickR * 10),           // Wick oranı
    ];
}

// ── Model Mimarisi ─────────────────────────────────────────────────────────
function _buildModel() {
    const m = tf.sequential();
    // Katman 1: Geniş özellik öğrenme
    m.add(tf.layers.dense({
        units: 64, activation: 'relu', inputShape: [14],
        kernelInitializer: 'glorotNormal',
        kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 })
    }));
    m.add(tf.layers.batchNormalization());
    m.add(tf.layers.dropout({ rate: 0.25 }));
    // Katman 2: Örüntü sıkıştırma
    m.add(tf.layers.dense({
        units: 32, activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 })
    }));
    m.add(tf.layers.dropout({ rate: 0.20 }));
    // Katman 3: Karar
    m.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    // Çıkış: 0 = bearish, 1 = bullish
    m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    m.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    return m;
}

// ── Kalıcı Hafıza Yükleme ve Kaydetme (IndexedDB & LocalStorage) ───────────
async function _loadModelFromStorage() {
    try {
        _mlModel = await tf.loadLayersModel('indexeddb://cyclemaster-ml-model');
        // Yüklenen modeli tekrar compile edelim (eğitim yapılabilmesi için)
        _mlModel.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Metadata yükle
        const savedAcc = localStorage.getItem('cyclemaster_ml_accuracy');
        if (savedAcc) _mlAccuracy = parseFloat(savedAcc);
        
        const savedTs = localStorage.getItem('cyclemaster_ml_train_ts');
        if (savedTs) _mlTrainTs = parseInt(savedTs, 10);
        
        _mlReady = true;
        console.log("🧠 ML: Kayıtlı sinir ağı başarıyla IndexedDB'den yüklendi.");
        return true;
    } catch (e) {
        console.log("🧠 ML: Kayıtlı model bulunamadı veya IndexedDB yüklenemedi. Yeni model oluşturulacak.");
        return false;
    }
}

async function _saveModelToStorage() {
    if (!_mlModel) return;
    try {
        await _mlModel.save('indexeddb://cyclemaster-ml-model');
        if (_mlAccuracy !== null) {
            localStorage.setItem('cyclemaster_ml_accuracy', _mlAccuracy.toString());
        }
        localStorage.setItem('cyclemaster_ml_train_ts', _mlTrainTs.toString());
        console.log("🧠 ML: Model ve ağırlıklar IndexedDB'ye başarıyla kaydedildi.");
    } catch (e) {
        console.warn("🧠 ML: Model kaydedilemedi:", e);
    }
}

// ── Arka Planda veya Bloklayarak Eğitim Motoru ────────────────────────────
async function _runTraining(closes, highs, lows, volumes, isBackground) {
    if (_mlTraining) return;
    _mlTraining = true;

    const n = closes.length;
    
    if (!isBackground) {
        _mlSetBadge('EĞİTİLİYOR...', 'b-gray', false);
        _mlSetStatus('🧠 ML modeli ilk kez eğitiliyor...');
    } else {
        _mlSetStatus('🧠 ML modeli arka planda güncelleniyor...');
    }

    try {
        // ── Veri seti oluştur ──────────────────────────────────────────────
        const featureRows = [];
        const labels      = [];
        const startIdx    = Math.max(31, n - 500);  // Son 500 bar

        let posCount = 0, negCount = 0;

        for (let i = startIdx; i < n - 3; i++) {
            const feat = _mlFeatures(closes, highs, lows, volumes, i);
            if (!feat) continue;

            // Etiket: 3 bar sonraki kümülatif getiri — eşik ile gürültüyü filtrele
            const ret3 = (closes[i + 3] - closes[i]) / (closes[i] || 1);
            // ATR bazlı dinamik eşik: küçük hareketleri "nötr" say, atla
            const atrArr = calcATR(
                highs.slice(0, i + 1), lows.slice(0, i + 1), closes.slice(0, i + 1), 14
            );
            const atr     = atrArr[atrArr.length - 1] || closes[i] * 0.005;
            const thresh  = atr / (closes[i] || 1);   // ATR / fiyat → dinamik eşik

            if (ret3 > thresh) {
                featureRows.push(feat);
                labels.push(1);    // LONG
                posCount++;
            } else if (ret3 < -thresh) {
                featureRows.push(feat);
                labels.push(0);    // SHORT
                negCount++;
            }
            // Nötr örnekler atlanıyor — gürültüyü eğitim setinden çıkar
        }

        // Sınıf dengesizliği kontrolü: fazla olan sınıfı kırp
        const minCount = Math.min(posCount, negCount);
        if (minCount > 5) {
            let pTrimmed = 0, nTrimmed = 0;
            const balancedFeats  = [];
            const balancedLabels = [];
            // Sondan başa doğru giderek en güncel (yeni) verileri koruyalım
            for (let j = featureRows.length - 1; j >= 0; j--) {
                if (labels[j] === 1 && pTrimmed < minCount) {
                    balancedFeats.unshift(featureRows[j]); // Kronolojik sırayı korumak için başa ekle
                    balancedLabels.unshift(1);
                    pTrimmed++;
                } else if (labels[j] === 0 && nTrimmed < minCount) {
                    balancedFeats.unshift(featureRows[j]);
                    balancedLabels.unshift(0);
                    nTrimmed++;
                }
            }
            featureRows.length = 0;
            labels.length      = 0;
            balancedFeats.forEach(f  => featureRows.push(f));
            balancedLabels.forEach(l => labels.push(l));
        }

        if (featureRows.length < 30) {
            _mlTraining = false;
            if (!isBackground) _mlSetBadge('VERİ AZ', 'b-gray', false);
            return;
        }

        const xs = tf.tensor2d(featureRows);
        const ys = tf.tensor2d(labels, [labels.length, 1]);

        // Transfer learning: mevcut model varsa fine-tune et, yoksa sıfırdan başla
        let trainModel;
        const isFineTune = !!_mlModel && _mlReady;
        if (isFineTune) {
            trainModel = _mlModel;   // Mevcut ağırlıkları koru → fine-tune
            // Fine-tune için daha düşük learning rate
            trainModel.compile({
                optimizer: tf.train.adam(0.0003),
                loss: 'binaryCrossentropy',
                metrics: ['accuracy']
            });
        } else {
            trainModel = _buildModel();   // Sadece ilk açılışta sıfırdan
        }

        let lastAcc = 0;
        const EPOCHS = isFineTune ? 5 : 45;   // Fine-tune'da çok daha az epoch yeterli (aşırı öğrenmeyi önler)

        await trainModel.fit(xs, ys, {
            epochs: EPOCHS,
            batchSize: 16,
            shuffle: false, // Zaman serisinde data leakage (veri sızıntısı) olmaması için false
            validationSplit: 0.15,
            verbose: 0,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    lastAcc = logs.acc || logs.accuracy || 0;
                    if (!isBackground && epoch % 5 === 0) {
                        const pct = Math.round(((epoch + 1) / EPOCHS) * 100);
                        _mlSetBadge('EĞİTİM ' + pct + '%', 'b-gray', false);
                    }
                }
            }
        });

        xs.dispose();
        ys.dispose();

        // Fine-tune'da eski modeli dispose etme — aynı referansı güncelliyoruz
        if (!isFineTune && _mlModel) {
            try { _mlModel.dispose(); } catch(e) {}
        }
        _mlModel    = trainModel;
        _mlAccuracy = lastAcc;
        _mlReady = true;
        _mlTrainTs = Date.now();

        // IndexedDB'ye kaydet
        await _saveModelToStorage();

        _mlSetStatus('🧠 ML modeli başarıyla eğitildi ve kaydedildi.');
        setTimeout(() => _mlSetStatus(''), 5000);

        // Eğer arka planda eğittiysek, bitince hemen yeni tahmini hesapla
        if (isBackground) {
            await _makePrediction(closes, highs, lows, volumes);
        }
    } catch (e) {
        console.error("🧠 ML eğitim hatası:", e);
        _mlSetStatus('🧠 ML eğitim hatası oluştu!');
    } finally {
        _mlTraining = false;
    }
}

// ── Tahmin Yürütme ve UI Güncelleme Yardımcısı ──────────────────────────────
async function _makePrediction(closes, highs, lows, volumes) {
    const n = closes.length;
    const lastFeat = _mlFeatures(closes, highs, lows, volumes, n - 1);
    if (!lastFeat || !_mlModel) return 50;

    const inp  = tf.tensor2d([lastFeat]);
    const pred = _mlModel.predict(inp);
    const val  = (await pred.data())[0];
    inp.dispose();
    pred.dispose();

    const mlPct  = Math.round(val * 100);
    _mlScore     = mlPct;
    const conf   = Math.abs(val - 0.5) * 200;   // 0-100 arası güven mesafesi
    const accTxt = _mlAccuracy ? ` Acc:${Math.round(_mlAccuracy*100)}%` : '';

    // Badge güncelle
    if (val > 0.62) {
        _mlDir = 'LONG';
        _mlSetBadge(`⬆ LONG ${mlPct}% ·${accTxt}`, 'b-green', true);
    } else if (val < 0.38) {
        _mlDir = 'SHORT';
        _mlSetBadge(`⬇ SHORT ${100 - mlPct}% ·${accTxt}`, 'b-red', true);
    } else {
        _mlDir = 'NÖTR';
        _mlSetBadge(`○ NÖTR ·${accTxt}`, 'b-gray', false);
    }

    // Confidence değerini global'e yaz (runAgent entegrasyonu için)
    window._mlConfidence = conf;

    // Agent reasoning kutusunu güncelle
    _mlUpdateReasoning();

    return mlPct;
}

// ── Ana Tahmin Fonksiyonu (async) ──────────────────────────────────────────
async function calcMLPrediction(highs, lows, closes, volumes) {
    // TF.js yüklü değilse
    if (typeof tf === 'undefined') {
        _mlSetBadge('TF YÜKLENİYOR', 'b-gray', false);
        return 50;
    }

    const n = closes.length;
    if (n < 60) {
        _mlSetBadge('VERİ YETERSİZ', 'b-gray', false);
        return 50;
    }

    // Zaten eğitiliyorsa, mevcut skorla hemen dön
    if (_mlTraining) return _mlScore;

    // Bellekte model yoksa IndexedDB'den yüklemeyi dene
    if (!_mlModel && !_mlReady) {
        const loaded = await _loadModelFromStorage();
        if (loaded) {
            // Hafızadan yüklendiyse hemen tahmini yapalım
            await _makePrediction(closes, highs, lows, volumes);
        }
    }

    // 30 dakikada bir yeniden eğit
    const now     = Date.now();
    const retrain = !_mlReady || (now - _mlTrainTs > 30 * 60 * 1000);

    if (retrain) {
        // Eğer zaten hazır bir modelimiz varsa (hafızadan yüklendiyse), 
        // eğitimi ARKA PLANDA başlatıp kullanıcıyı bekletmeyelim.
        if (_mlReady) {
            _runTraining(closes, highs, lows, volumes, true); // non-blocking arka plan eğitimi
        } else {
            // Model hiç yoksa (ilk açılışta ve kayıt yoksa), bloklayarak eğit
            await _runTraining(closes, highs, lows, volumes, false);
        }
    }

    // Tahmin yürüt (mevcut model ile)
    if (_mlModel && _mlReady) {
        return await _makePrediction(closes, highs, lows, volumes);
    }

    return 50;
}

// ── Agent Reasoning Entegrasyonu ───────────────────────────────────────────
function _mlUpdateReasoning() {
    const rBox = document.getElementById('agent-reasoning');
    if (!rBox || !_mlReady) return;

    // Önceki ML satırını kaldır (duplicate önle)
    const existing = rBox.querySelector('.ml-reasoning-line');
    if (existing) existing.remove();

    const dir  = _mlDir;
    const pct  = _mlScore;
    const conf = window._mlConfidence || 0;
    const acc  = _mlAccuracy ? ` · Doğruluk: ${Math.round(_mlAccuracy * 100)}%` : '';
    const clr  = dir === 'LONG' ? 'var(--green)' : dir === 'SHORT' ? 'var(--red)' : '#a1a1aa';
    const icon = dir === 'LONG' ? '⬆' : dir === 'SHORT' ? '⬇' : '○';

    // Aktif yönü al (index.html'deki lastMode ile çapraz kontrol)
    const sysDir = typeof lastMode !== 'undefined' ? lastMode : null;
    const conflict = sysDir && dir !== 'NÖTR' && sysDir !== 'YATAY' && dir !== sysDir;
    const strongSignal = conf >= 45;   // %62.5+ veya %37.5- güven eşiği

    let extraHtml = '';
    if (conflict && strongSignal) {
        extraHtml = `<div style="margin-top:4px;padding:4px 8px;border-radius:6px;`
            + `background:rgba(244,63,94,0.12);border:1px solid rgba(244,63,94,0.3);`
            + `font-size:0.7rem;color:#f43f5e;font-weight:600;">`
            + `⚠️ VETO: Derin Öğrenme sisteme <strong>KARŞI</strong> yön gösteriyor — `
            + `giriş öncesi ekstra teyit alın!</div>`;
    } else if (!conflict && dir !== 'NÖTR' && strongSignal) {
        extraHtml = `<div style="margin-top:4px;font-size:0.68rem;color:#10b981;">` 
            + `✓ Derin Öğrenme sistem yönünü destekliyor</div>`;
    }

    const line = document.createElement('div');
    line.className = 'ml-reasoning-line';
    line.innerHTML = '<hr style="border-color:rgba(255,255,255,0.08);margin:6px 0">'
        + `<span style="color:${clr};font-weight:700">🧠 Derin Öğrenme: ${icon} ${dir} (${pct}%${acc})</span>`
        + extraHtml;
    rBox.appendChild(line);
}

// ── Global erişim ──────────────────────────────────────────────────────────
window.calcMLPrediction   = calcMLPrediction;
window._mlUpdateReasoning = _mlUpdateReasoning;
window._mlGetScore        = () => _mlScore;
window._mlGetDir          = () => _mlDir;
window._mlGetConfidence   = () => window._mlConfidence || 0;
window._mlIsReady         = () => _mlReady;
