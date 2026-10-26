
## 2026-10-10T19:51:56+00:00
DeepSurv macOS OpenMP kilitlenmesi nedeniyle izole alt süreçte çalıştırılıyor; 30 dk zaman aşımı eklendi. Model, ızgara ve seçim kuralı değişmedi. Henüz tam geliştirme koşusu yapılmadı.
Changed files: src/biobridge/deepsurv_worker.py, src/biobridge/experiment.py, src/biobridge/external.py, src/biobridge/models.py

## 2026-10-11T15:08:35+00:00
Keşifsel analiz eklendi: Witowski 2026 tarzı eşit ağırlıklı XGBoost füzyonu (F2: AFT genler+modüller; F4: AFT+Cox genler+modüller). Plan 08_exploratory_fusion.py içinde sabit; yeniden eğitim yok; sonuçlar bu kayıttan sonra hesaplanacak.
Changed files: -

## 2026-10-11T15:11:55+00:00
Keşifsel analiz eklendi: Witowski 2026 tarzı eşit ağırlıklı XGBoost füzyonu (F2: AFT genler+modüller; F4: AFT+Cox genler+modüller). Plan 08_exploratory_fusion.py içinde sabit; yeniden eğitim yok; sonuçlar bu kayıttan sonra hesaplanacak.
Changed files: scripts/08_exploratory_fusion.py

## 2026-10-11T16:57:24+00:00
Makale analizleri eklendi: öğrenme eğrisi (H1, H2 önceden yazılı), klinik değer, SHAP, şekiller. Planlar betiklerin içinde sabit; sonuçlar bu kayıttan sonra hesaplanacak.
Changed files: scripts/09_learning_curve.py, scripts/10_clinical_utility.py, scripts/11_shap.py, scripts/12_figures_tables.py, src/biobridge/paper.py

## 2026-10-11T17:41:02+00:00
Son paket: keşifsel kesişim yeniden kalibrasyonu (13), ek tablolar (14), Şekil 1 (15), yeni kohort doğrulayıcı analizi (16; birincil hipotez F2 vs XGB-AFT genler, kohort verisi görülmeden önce eklendi).
Changed files: scripts/13_recalibration.py, scripts/14_supplementary_tables.py, scripts/15_figure1_design.py, scripts/16_new_cohort_confirmatory.py

## 2026-10-11T17:46:30+00:00
Şekil düzenlemesi (12): yalnızca görsel değişiklikler; sıralama, etiketler, öğrenme eğrisine fark paneli (C). Analizlerde değişiklik yok.
Changed files: scripts/12_figures_tables.py

## 2026-10-11T18:39:13+00:00
Ek karşılaştırma modelleri (birincil analizden sonra): 5 derin öğrenme modeli (DeepHit, Logistic-Hazard, N-MTLR, PC-Hazard, Neural AFT); DeepSurv ile aynı birimler, girdiler, ızgara ve erken durdurma; temel risk başlatması; hipotezleri değiştirmez.
Changed files: scripts/17_additional_dl.py, src/biobridge/dl_worker.py
