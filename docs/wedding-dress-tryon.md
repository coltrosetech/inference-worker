# Gelinlik prova (wedding dress try-on)

Kişinin mevcut kıyafetini **bir gelinlik referansıyla** değiştiren güvenli
virtual try-on reçetesi. `tryon` preset'i kullanılır; çıktı daima giyiniktir
ve çıplaklık `SAFETY_NEGATIVE` ile kalıcı olarak negative prompt'a gömülüdür
(çağıran kaldıramaz).

## Girdi gereksinimleri

- **input_image** — kişinin **tam boy ya da en az üst-gövde** fotoğrafı.
  Yakın plan yüz fotoğrafı işe yaramaz: değiştirilecek bir giysi bölgesi olmalı.
- **reference_image** — giydirilecek **gelinlik** görseli (ürün/manken/kişi fotoğrafı).
- mask — verilmezse `auto_mask=true` ile SegFormer mevcut kıyafeti otomatik maskeler.

## Önerilen parametreler

| Parametre | Değer | Neden |
|---|---|---|
| `auto_mask` | `true` | Mevcut kıyafeti otomatik maskele |
| `auto_mask_categories` | `["upper_clothes", "dress", "skirt", "pants", "belt"]` | Gelinlik tam boy olduğundan tüm mevcut kıyafet bölgesi maskelenir |
| `grow_mask_px` | `20` | Gelinlik kenarları için yumuşak geçiş |
| `reference_weight` | `0.9` | Referans gelinliğe güçlü bağlılık (IP-Adapter) |
| `strength` | `0.92` | Maskeli bölgeyi yeniden boya |
| `steps` | `28` | Detay/süre dengesi (RTX 5090'da ~6-8 s) |
| `cfg` | `7.0` | JuggernautXL Inpaint için uygun |

Negative prompt'a ek bir şey yazmana gerek yok; çıplaklık-dışlama otomatik eklenir.

## Örnek istek

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H "Authorization: Bearer $WORKER_API_KEY" \
  -H "Idempotency-Key: $(uuidgen)" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "wd_demo_001",
    "preset": "tryon",
    "prompt": "wearing an elegant white lace A-line wedding gown, long flowing skirt, fitted bodice, natural studio lighting, photorealistic",
    "input_image_url": "https://your-storage/person_fullbody.jpg",
    "reference_image_url": "https://your-storage/wedding_dress.jpg",
    "parameters": {
      "auto_mask": true,
      "auto_mask_categories": ["upper_clothes", "dress", "skirt", "pants", "belt"],
      "grow_mask_px": 20,
      "reference_weight": 0.9,
      "strength": 0.92,
      "steps": 28,
      "cfg": 7.0
    },
    "callback_url": "https://your-backend/callbacks/wd_demo_001",
    "upload_url": "https://your-storage/signed-put-url"
  }'
```

## Güvence

- Hedef giysi (reference_image) **zorunlu** — "boş/çıplak" hedef yolu yoktur.
- `nude, naked, undressed, lingerie, nsfw...` negative'e sabit gömülü; çıktı giyinik.
- Yüz/saç/arka plan auto-mask ile korunur; yalnızca giysi bölgesi değişir.
