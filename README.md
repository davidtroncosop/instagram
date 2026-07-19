# Instagram affiliate video pipeline

MVP en Python para:

1. poner una prenda sobre un modelo con `gpt-image-2`;
2. convertir esa imagen en video con `gemini-omni-flash-preview`;
3. editar un video base conservando su movimiento (`--base-video`);
4. publicar opcionalmente el MP4 como Reel mediante Instagram Graph API.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Completa las claves en `.env`:

- `OPENAI_API_KEY`: clave de OpenAI.
- `GEMINI_API_KEY`: clave de Google AI Studio/Gemini.
- `INSTAGRAM_ACCESS_TOKEN`: token con permiso de publicación.
- `INSTAGRAM_USER_ID`: ID de la cuenta profesional de Instagram.
- `META_API_VERSION`: versión habilitada en tu aplicación de Meta.

No compartas ni subas `.env` a git.

## Entradas

- `modelo.jpg`: persona/modelo con derechos para usar la imagen.
- `prenda.png`: foto de la ropa. Una imagen con fondo limpio funciona mejor.
- `mask.png`: opcional. Debe tener el mismo tamaño/formato que `modelo.jpg`, incluir canal alpha y marcar la zona que se editará. Si se usa con dos imágenes, la máscara se aplica a la primera (`modelo.jpg`).
- `movimiento.mp4`: opcional. Si se entrega, Gemini intenta conservar su pose, cámara y movimiento.

## Uso

Solo imagen → video:

```bash
python pipeline.py \
  --model-image modelo.jpg \
  --garment-image prenda.png
```

Video-to-video:

```bash
python pipeline.py \
  --model-image modelo.jpg \
  --garment-image prenda.png \
  --mask mask.png \
  --base-video movimiento.mp4
```

Los archivos quedan en `outputs/`.

## Publicar en Instagram

Instagram no puede leer un archivo local. Primero sube el MP4 a Cloudflare R2, Amazon S3, Google Cloud Storage u otro hosting con una URL HTTPS pública. Después:

```bash
python pipeline.py \
  --model-image modelo.jpg \
  --garment-image prenda.png \
  --base-video movimiento.mp4 \
  --publish \
  --public-url "https://tu-dominio.com/reel.mp4" \
  --caption "¿Te gusta este outfit? Comenta LINK y te lo envío por DM."
```

El script crea el contenedor de Reel, espera `FINISHED` y llama a `media_publish`.

## ManyChat

ManyChat se configura aparte, no dentro de Python:

1. Conecta la cuenta profesional de Instagram.
2. Crea `Auto-DM Links from comments`.
3. Selecciona el Reel y la palabra clave `LINK`.
4. Responde públicamente que enviarás el enlace.
5. Envía un Opening DM con un botón y después el enlace afiliado.
6. Usa un enlace distinto por Reel o un webhook si quieres resolver enlaces dinámicamente.

Incluye la divulgación de afiliado en el caption y/o mensaje, y activa la etiqueta de colaboración pagada cuando corresponda.

## Fuentes

- [OpenAI: edición y referencias con GPT Image 2](https://developers.openai.com/api/docs/guides/image-generation#edit-images)
- [Google: Gemini Omni Flash para video](https://ai.google.dev/gemini-api/docs/omni)
- [Meta: publicación de Reels](https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-8d93f052-4c50-4cef-b23e-57732bf370f3)
- [ManyChat: Auto-DM desde comentarios](https://help.manychat.com/hc/en-us/articles/16654065283100)
