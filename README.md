# Instagram affiliate video pipeline

Pipeline en Python para:

1. fusionar una foto de modelo con hasta nueve vistas de una prenda usando `gpt-image-2`;
2. convertir esa imagen en video con `gemini-omni-flash-preview`;
3. editar un video base conservando su movimiento (`--base-video`);
4. publicar opcionalmente el mismo MP4 en Instagram y TikTok.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Completa `.env` según el backend que quieras usar:

- `OPENAI_API_KEY`: clave de OpenAI.
- `GEMINI_BACKEND=vertex`: usa Gemini Omni Flash desde Google Cloud Agent Platform.
- `GOOGLE_CLOUD_PROJECT`: ID del proyecto de GCP con facturación y Agent Platform API.
- `GOOGLE_CLOUD_LOCATION=global`: ubicación de Gemini Omni Flash.
- `GEMINI_GCS_BUCKET`: bucket `gs://` opcional para subir los medios de entrada de Vertex; se recomienda para videos locales grandes.
- `GEMINI_API_KEY`: solo se usa si `GEMINI_BACKEND=ai_studio`.
- `INSTAGRAM_ACCESS_TOKEN`: token con permiso de publicación.
- `INSTAGRAM_USER_ID`: ID de la cuenta profesional de Instagram.
- `META_API_VERSION`: versión habilitada en tu aplicación de Meta.
- `CLOUDINARY_URL`: opcional; si está configurado, `--publish` sube automáticamente
  el MP4 como video público y usa la URL HTTPS devuelta por Cloudinary.
- `TIKTOK_ACCESS_TOKEN`: token de usuario con el scope `video.publish` para publicación directa.
- `TIKTOK_PRIVACY_LEVEL`: nivel permitido por la cuenta, normalmente `PUBLIC_TO_EVERYONE`.
- `GEMINI_ASPECT_RATIO=9:16`: formato vertical del video.
- `GEMINI_VIDEO_DURATION=10s`: duración solicitada; Gemini Omni Flash admite entre 3 y 10 segundos.
- `MODEL_IMAGE_PATH` y `BASE_VIDEO_PATH`: rutas opcionales para reemplazar los archivos predeterminados de `Descargas`.
- `GARMENT_DIR`: carpeta opcional para las fotos de la prenda; por defecto es `Descargas/prendas`.
- `FISH_AUDIO_API_KEY`, `FISH_AUDIO_MODEL=s2.1-pro-free`, `FISH_AUDIO_VOICE_ID` y
  `FISH_AUDIO_API_URL`: se usan con `--voiceover`. El código también acepta los
  nombres antiguos `FISH_API_KEY`, `FISH_REFERENCE_ID` y `FISH_TTS_URL`.
- `GROQ_API_KEY`: opcional; se usa con `--subtitles` para obtener marcas de tiempo por palabra.

No compartas ni subas `.env` a git.

Para usar Agent Platform localmente, configura Application Default Credentials:

```bash
gcloud auth application-default login
```

El endpoint de Cloud requiere autenticación OAuth/ADC; una clave de AI Studio
no activa el crédito de GCP.

## Entradas

- `modelo.jpg`: persona/modelo con derechos para usar la imagen.
- `prendas/`: una o más imágenes de la misma prenda. Para cuatro vistas, usa cuatro archivos PNG/JPG/WEBP.
- `mask.png`: opcional. Debe tener el mismo tamaño/formato que `modelo.jpg`, incluir canal alpha y marcar la zona que se editará. Si se usa con dos imágenes, la máscara se aplica a la primera (`modelo.jpg`).
- `movimiento.mp4`: opcional. Si se entrega, Gemini intenta conservar su pose, cámara y movimiento.

## Uso

Solo imagen → video:

```bash
python pipeline.py \
  --model-image modelo.jpg \
  --garment-dir prendas
```

Video-to-video:

```bash
python pipeline.py \
  --model-image modelo.jpg \
  --garment-image prenda-frente.png \
  --garment-image prenda-espalda.png \
  --garment-image prenda-lateral-a.png \
  --garment-image prenda-lateral-b.png \
  --mask mask.png \
  --base-video movimiento.mp4
```

Con los archivos que tienes en `Descargas`, cuando agregues las cuatro vistas
de la prenda a `Descargas/prendas/`, puedes ejecutar directamente:

```bash
python pipeline.py
```

También puedes indicar otra carpeta explícitamente con
`--garment-dir "$HOME/Downloads/prendas"`.

Si quieres obtener las cuatro fotos automáticamente, copia la URL de la ficha
oficial de Falabella encontrada desde Knasta y usa:

```bash
python pipeline.py \
  --product-url "https://www.falabella.com/falabella-cl/product/..."
```

Knasta se usa para descubrir y comprobar la oferta; las imágenes se descargan
desde la ficha oficial de Falabella y quedan guardadas en `outputs/` junto con
un `source.json`. No se descarga la base de imágenes de Knasta.

Ya quedó preparado en este proyecto un lote de prueba real:
`garments/diadora-poleron/` contiene cuatro vistas del polerón Diadora y
`offers/diadora-poleron.json` contiene la oferta de $24.990 a $19.990. El
guion de prueba, sin afirmar stock ni tallas, sería:
`Mira lo que encontré... polerón deportivo Diadora, de 24 mil novecientos noventa bajó a 19 mil novecientos noventa. Comenta LOOK y te mando el link...`

Para usar ese lote:

```bash
python pipeline.py \
  --model-image "$HOME/Downloads/ChatGPT_Image_22_jul_2026,_202607252032.jpeg" \
  --garment-dir garments/diadora-poleron \
  --base-video "$HOME/Downloads/0718_202607252032.mp4" \
  --organic-test \
  --narration-text "Mira lo que encontré... polerón deportivo Diadora, de 24 mil novecientos noventa bajó a 19 mil novecientos noventa. Comenta LOOK y te mando el link..." \
  --voiceover \
  --subtitles
```

Para probar el contenido antes de tener Creators F, agrega `--organic-test`.
Ese modo no declara enlaces afiliados ni comisiones:

```bash
python pipeline.py \
  --organic-test \
  --narration-text "Mira esta oferta de moda y sígueme para más precios verificados." \
  --voiceover \
  --subtitles
```

Cuando quieras publicar esa prueba orgánica en Instagram, agrega `--publish`
y una URL HTTPS pública con `--public-url`. No uses todavía un enlace de
Creators F en este modo.

El script usa automáticamente:

- `Descargas/ChatGPT_Image_22_jul_2026,_202607252032.jpeg` como modelo y fondo.
- `Descargas/0718_202607252032.mp4` como video base.

El prompt predeterminado reemplaza a la chica del video por la chica de la
foto, conserva el ambiente de la foto y viste la prenda usando todas sus
vistas. Puedes sustituirlo con `--outfit-prompt` y `--video-prompt`.

## Voz y subtítulos opcionales

Para generar voz con Fish Audio y subtítulos sincronizados con Groq:

```bash
python pipeline.py \
  --garment-dir "$HOME/Downloads/prendas" \
  --narration-text "Mira esta oferta y comenta LOOK para recibir el enlace." \
  --voiceover \
  --subtitles
```

También puedes generar el guion desde un archivo JSON de oferta:

```bash
python pipeline.py \
  --garment-dir "$HOME/Downloads/prendas" \
  --offer-json oferta.json \
  --generate-script \
  --voiceover \
  --subtitles
```

El video generado por Gemini Omni Flash tiene un límite de 10 segundos. Por
eso el guion comercial se limita a 28 palabras y sigue un formato breve como:
`Mira lo que encontré... polerón oversais, de 35 mil bajó a 12 mil dos cincuenta. Quedan pocas tallas. Comenta LOOK y te mando el link...`

Si ya tienes una imagen de la persona que quieres usar como reemplazo, puedes
saltarte GPT Image 2 y usarla directamente como referencia de Gemini:

```bash
python pipeline.py \
  --reference-image persona.jpg \
  --base-video movimiento.mp4 \
  --video-prompt "Recreate video replace girl"
```

En este modo, `--reference-image` es la nueva persona y `--base-video` es el
video cuyo movimiento se quiere conservar.

Los archivos quedan en `outputs/`.

Gemini Omni Flash genera videos de 3 a 10 segundos y acepta videos de entrada
de hasta 10 segundos. El valor predeterminado del pipeline es `GEMINI_VIDEO_DURATION=10s`.

## Publicar en Instagram y TikTok

Instagram y TikTok no pueden leer un archivo local. Si `CLOUDINARY_URL` está
configurado, el pipeline sube automáticamente el MP4 y usa la URL HTTPS pública
devuelta por Cloudinary. También puedes entregar una URL manual con
`--public-url`. Para TikTok con `PULL_FROM_URL`, el dominio debe estar verificado
en tu aplicación de TikTok. Ejemplo:

```bash
python pipeline.py \
  --model-image modelo.jpg \
  --garment-dir prendas \
  --base-video movimiento.mp4 \
  --publish-both \
  --caption "¿Te gusta este outfit? Comenta LINK y te lo envío por DM."
```

`--publish` publica solo en Instagram; `--publish-tiktok` publica solo en TikTok;
`--publish-both` usa el mismo MP4 y caption en ambas plataformas. El script crea
el contenedor de Reel, espera `FINISHED` y llama a `media_publish`; para TikTok
inicia el Direct Post y consulta su estado hasta `PUBLISH_COMPLETE`.

La publicación directa en TikTok requiere una app registrada, el producto
Content Posting API, autorización del scope `video.publish` y autorización de
la cuenta creadora. Los clientes no auditados pueden quedar restringidos a
publicaciones privadas. Revisa la [documentación oficial de TikTok](https://developers.tiktok.com/doc/content-posting-api-get-started//).

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
- [Google Cloud: Gemini Omni Flash en Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/omni-flash-preview?hl=es)
- [Google Cloud: generación de video con Gemini Omni Flash](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/generate-videos-from-text)
- [Fish Audio: Text to Speech](https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech)
- [Groq: Speech to Text](https://console.groq.com/docs/speech-to-text)
- [MoviePy: documentación](https://zulko.github.io/moviepy/)
- [Knasta: comparación de precios](https://knasta.cl/)
- [Knasta: términos de uso](https://knasta.cl/terms.html)
- [Meta: publicación de Reels](https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-8d93f052-4c50-4cef-b23e-57732bf370f3)
- [TikTok: Content Posting API](https://developers.tiktok.com/doc/content-posting-api-get-started//)
- [ManyChat: Auto-DM desde comentarios](https://help.manychat.com/hc/en-us/articles/16654065283100)
