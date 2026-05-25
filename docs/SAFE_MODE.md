# SAFE MODE — projenin güvenli moda çekilmesi (değişiklik notları)

Bu doküman, inference-worker'ın **rızasız soyma (nudify) yeteneğinden arındırılıp**
yalnızca **kıyafet değiştirme (try-on) + video animasyon** yapan güvenli bir ürüne
dönüştürülmesi sürecinde yapılan **tüm değişiklikleri dosya-dosya** kaydeder. Amaç:
ileride prod'a taşırken neyin neden böyle olduğunu bilmek ve sınırı korumak.

> Tarih: 2026-05-24

---

## 1. Güvenlik sınırı (değişmez kural)

**Bu sistem kıyafeti yalnızca *değiştirebilir*, *çıkaramaz*. Çıktı her zaman giyiniktir.**

- Giysi-çıkarma / çıplak vücut üretme / "kimliği koru + soy" mekanizması **yasak** —
  "fictional karakter" ya da herhangi bir çerçeveyle bile geri getirilmez. Tehlike
  bu mekanizmanın kendisinde gömülüdür (NCII'nin tanımlayıcı yöntemi).
- Meşru iş: kıyafet değiştirme (output giyinik), kalite iyileştirme, video animasyon.

---

## 2. KALDIRILANLAR (soyma / vücut-üretim motoru)

Bunlar tamamen silindi — geri eklenmemeli:

**Preset dosyaları (silindi):**
- `worker/presets/fictional_character_body_reconstruction.py` — varsayılan prompt'u
  açıkça `"completely nude, photorealistic human body"` olan nudify preset'i.
- `worker/presets/fictional_recon.py` — aynı amaçlı "clean foundation" varyantı.
- `worker/presets/inpaint.py` — çekirdek inpaint motoru; içinde **undress→redress**
  iki-geçiş mantığı vardı (`UNDRESS_NEGATIVE`, `DEFAULT_SKIN_PROMPT="bare natural skin,
  torso, arms, body, anatomy"`, `_two_pass_manual` denoise=1.0 ile çıplak tene indiriyordu).
- `worker/presets/inpaint_sdxl.py`, `worker/presets/inpaint_realvis.py` — inpaint'i miras
  alan "quality pack"; `bust_emphasis` + `lora_curvy`/`lora_skin`/`lora_anatomy` vücut-
  gerçekçiliği yığını.
- `worker/presets/inpaint_premium.py`, `edit.py`, `edit_premium.py`, `style.py`,
  `controlnet.py` — kapsam dışı bırakıldı (yalnızca tryon hedeflendi).

**Workflow'lar (silindi):** `workflows/` altındaki `fictional_*.json`,
`inpaint*.json`, `edit*.json`, `style.json`, `controlnet.json`.

**Diğer:** `docs/fictional_character_body_reconstruction.md`,
`scripts/validate_fictional_body_recon.py`.

**Testler (silindi):** `tests/test_presets/test_{controlnet,edit,edit_premium,
inpaint,inpaint_premium,inpaint_sdxl_realvis,style,ltx_video}.py`.

---

## 3. GÜVENLİK GUARD'LARI (eklenen / değiştirilen)

### `worker/presets/tryon.py` (tek görsel preset'i; kendi kendine yeten)
- **`SAFETY_NEGATIVE`** sabiti: `nude, nudity, naked, undressed, topless, bottomless,
  bare skin, exposed breasts, exposed genitals, lingerie, underwear, sexual, nsfw`.
  Her işin negative prompt'una **çağıran kaldıramayacak şekilde** öne eklenir
  (`inject` içinde `negative = f"{SAFETY_NEGATIVE}, {params.negative_prompt}"`).
- **Hedef giysi zorunlu:** `reference_image` olmadan `ValueError` — "boş/çıplak hedef"
  yolu yok. Maske her zaman *yeni bir giysiyle* doldurulur.
- **Tek geçiş:** undress→redress iki-geçiş mantığı yok.
- Paylaşılan yardımcılar (`random_seed`, `ALLOWED_CATEGORIES`, `DEFAULT_AUTO_CATEGORIES`)
  silinen modüllerden buraya **inline** edildi.
- Kalite: opsiyonel **hires-fix** (4x-UltraSharp upscale → düşük-denoise resample),
  varsayılan açık.

### `worker/presets/__init__.py`
- `PRESETS` registry yalnızca güvenli preset'leri içerir: `tryon`, `ltx_video`, `wan_flf2v`.
- Tüm `fictional_*` / `inpaint*` / `edit*` / `style` / `controlnet` importları kaldırıldı.

### `worker/api/generate.py`
- `preset: Literal["tryon", "ltx_video", "wan_flf2v"]` — başka preset adı 422 ile reddedilir.
- `tryon` için `reference_image_url` **zorunlu** (yoksa 400); `wan_flf2v` için END frame
  (`reference_image_url`) zorunlu. Silinen preset'lere ait koşullar temizlendi.

### `webapp/main.py`
- `/api/generate` allowlist: `{"tryon", "ltx_video", "wan_flf2v"}`.
- Video çıktı tipi (`ext`, `output_kind`) hem `ltx_video` hem `wan_flf2v` için `video/mp4`.
- **Yeni:** `/api/progress` endpoint'i — ComfyUI kuyruğu + log'daki tqdm satırını okuyup
  `{running, step, total, detail}` döner (canlı "step X/Y" göstergesi için).

### `worker/main.py`
- `_warmup_and_mark_ready`: tüm preset'ler (`tryon`, `ltx_video`, `wan_flf2v`) başarıyla
  warm olunca `mark_ready()` (`if results and all(results.values())`). Üçü de geçmezse
  worker `ready:false` kalır — yani bir preset'in workflow'u bozulursa erkenden yakalanır.

### Frontend (`webapp/frontend/src/`)
- `lib/api.ts`: `Preset` tipi yalnızca `"tryon" | "ltx_video" | "wan_flf2v"`. `SamplingProgress`
  tipi + `getProgress()` eklendi.
- `components/PresetRail.tsx`: seçici yalnızca güvenli preset kartlarını gösterir
  (tüm `fictional_*`/`inpaint*`/`edit*` kartları silindi).
- `components/PresetForm.tsx`: baştan tryon-only yazıldı; sonra `ltx_video` ve `wan_flf2v`
  dalları eklendi. Eski nudify form alanları (`bust_emphasis`, `anatomy_boost`,
  `aggressive_body_mask`, `two_pass`, `skin_prompt`, LoRA/detailer knob'ları, nude varsayılan
  prompt'lar) **tamamen kaldırıldı**.
- `components/OutputViewer.tsx`: iş çalışırken **canlı "step X/Y" + ilerleme çubuğu**
  (`/api/progress` polling).
- `App.tsx`: varsayılan preset `tryon`; preset-bazlı girdi gereksinimleri (tryon: ref+mask,
  ltx_video: yok, wan_flf2v: ref=END frame).

### Testler
- `test_registry.py`, `test_tryon.py`, `test_api/test_generate.py`, `test_api/test_cancel.py`
  güvenli preset setine göre güncellendi. Undress/two-pass testleri silindi; yerine
  "bu alanlar artık reddedilmeli" ve "SAFETY_NEGATIVE gömülü" testleri eklendi.

---

## 4. Güvenli mimari — şu anki preset'ler

| Preset | Tür | Ne yapar | Güvenlik |
|---|---|---|---|
| `tryon` | image | Kıyafet **değiştirme** (referans giysi + auto-mask + IP-Adapter + hires-fix) | Çıktı giyinik; SAFETY_NEGATIVE gömülü; referans zorunlu |
| `ltx_video` | video | Tek görseli (giyinik try-on çıktısı) canlandırır (LTX-Video 13B) | Yalnızca verilen görseli animasyonlar; kıyafeti değiştirmez |
| `wan_flf2v` | video | start→end geçişi (Wan 2.1 FLF2V 14B + LightX2V hız LoRA'sı) | İki giyinik kare arası interpolasyon |

---

## 5. PROD-SAFE'E GİDERKEN — yapılması önerilenler (kalan işler)

Bunlar henüz yapılmadı; safe modu sağlamlaştırmak için sıradaki adımlar:

1. **Eski doküman/iz temizliği:**
   - `CLAUDE.md` içindeki `"Minimal-filter models are preferred"` satırı ve `fictional_*`
     atıfları kaldırılmalı.
   - `docs/quality-pack.md`, `docs/admin-panel.md`, `docs/CLAUDE.md.v2026-05-03-sanitized.md`
     eski nudify yığınına atıf yapıyorsa güncellenmeli/silinmeli.
   - `.env`'de `WORKER_API_KEY` **iki kez** tanımlı (satır 3 ve 27) — birini sil.
2. **Girdi provenance / kötüye kullanım önleme (prod için kritik):**
   - Yüklenen görsellerde **yaş/rıza** kontrolü (politika + opsiyonel sınıflandırıcı).
   - Çıktıya **C2PA / görünmez watermark** ("AI-generated" işareti) + üretim loglama.
   - Oran sınırlama, kimlik doğrulama, ToS onayı.
3. **NSFW çıktı sınıflandırıcısı:** üretilen görüntü/video bir NSFW dedektöründen geçsin;
   SAFETY_NEGATIVE'e ek ikinci savunma katmanı.
4. **Kod seviyesi kilit:** SAFETY_NEGATIVE ve "reference zorunlu" kuralları için
   regresyon testleri (zaten kısmen var) genişletilsin; CI'da zorunlu kılınsın.
5. **Gözlemlenebilirlik:** her işin preset/parametrelerini denetim için logla.

---

## 6. Ops / runtime notları

- Worker: `uvicorn worker.main:app` :8000 (manuel/setsid başlatılıyor).
- Webapp: `uvicorn webapp.main:app` :8001; frontend `webapp/frontend/dist` (Vite build).
- ComfyUI: supervisor programı `comfyui` :18188 →
  `supervisorctl -c /etc/supervisor/supervisord.conf restart comfyui`.
- Public erişim: cloudflare quick-tunnel → :8001 (`*.trycloudflare.com`; pid yeniden
  başlarsa adres değişir → kalıcı için adlandırılmış tünel).
- Modeller: LTX 13B distilled, Wan 2.1 FLF2V 14B fp8 + umt5 + clip_vision + LightX2V LoRA,
  RealVis/Juggernaut inpaint + IP-Adapter + 4x-UltraSharp.
- Custom node'lar: ComfyUI native Wan (FLF2V dahil) + LTXVideo + VideoHelperSuite +
  IPAdapter_plus.

---

## 7. Özet
Soyma motoru (preset'ler + iki-geçiş + nude prompt'lar + vücut LoRA'ları) **kökten kaldırıldı**;
yerine kıyafet-değiştirme + video, **kaldırılamayan çıplaklık-dışlama negatifi** ve
**zorunlu giysi referansı** ile güvenli bir akış kuruldu. Prod'a taşımadan önce §5'teki
provenance/sınıflandırıcı/politika katmanları eklenmelidir.
