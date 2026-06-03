# CGM to PNG Preview Feasibility (WSL)

## Conclusion

CGM-to-PNG preview generation is **not feasible in the current WSL environment without installing additional conversion tooling**.

Evidence:
- `chroma_db/assets_manifest.json` contains found CGM assets, including brake-related assets.
- The usual command-line converters checked for this task are not present in `PATH`: ImageMagick (`magick`/`convert`), GraphicsMagick (`gm`), `rsvg-convert`, Inkscape, LibreOffice/soffice, and common CGM-specific converter command names.
- The only available image library found during this investigation, Python Pillow, cannot identify the binary CGM samples and produced no PNG files.

## Sample assets tested

Manifest inspection found 44 `found` CGM asset references, 39 unique CGM asset paths, among 54 found assets total. The following brake-related CGM files were selected from `chroma_db/assets_manifest.json` and verified on disk under `docs/S1000D Issue 6/Bike Data Set for Release number 6 R2/`:

| Asset | Manifest title | Size | `file` result | SHA-256 |
|---|---|---:|---|---|
| `ICN-C0419-S1000D0379-001-01.CGM` | Cantilever brake with straddle cable | 80,630 bytes | binary Computer Graphics Metafile, version 4, parameter length 28 | `3813c57bfdb3a67353be5c38d49f4d044a82c13faa65fcf5746c22232f5ab0a6` |
| `ICN-C0419-S1000D0380-001-01.CGM` | Exploded diagram of a brake | 57,298 bytes | binary Computer Graphics Metafile, version 4, parameter length 28 | `978f6a6bdfde9ece2bb3b8aedfa2515cdc6657aedf54f930d86a14c481bdb842` |
| `ICN-C0419-S1000D0382-001-01.CGM` | Brake pad seating | 81,652 bytes | binary Computer Graphics Metafile, version 4, parameter length 28 | `547fd98b8736b4efd5f152a6e3c7dd58ac53b71c22944d0c284882277cca7a1b` |

## Available tools checked

| Tool / library | Availability | Notes |
|---|---|---|
| `magick` | Not found | ImageMagick CLI unavailable. |
| `convert` | Not found | ImageMagick legacy CLI unavailable. |
| `gm` | Not found | GraphicsMagick unavailable. |
| `rsvg-convert` | Not found | SVG converter unavailable and not a direct CGM converter. |
| `inkscape` | Not found | Inkscape unavailable. |
| `libreoffice` | Not found | LibreOffice unavailable. |
| `soffice` | Not found | LibreOffice headless CLI unavailable. |
| `cgm`, `cgm2png`, `cgm2svg`, `cgm2pdf`, `cgm2ps`, `cgmview`, `cgmtopng`, `uniconvertor` | Not found | Common CGM-related converter names unavailable in `PATH`. |
| `file` | `/usr/bin/file` | Available for format verification only. |
| `identify` | Not found | ImageMagick identify unavailable. |
| Python `PIL` / Pillow | Available | Does not decode these CGM files. |
| Python `wand`, `cairosvg`, `matplotlib`, `cv2` | Not available | Not usable for conversion in this environment. |

## Conversion attempts

All conversion outputs were directed to `/tmp/jarvis-runtime/s1000d-cgm-preview-test`.

| Attempt | Exact command | Exit code | stdout summary | stderr summary | PNG produced? |
|---|---|---:|---|---|---|
| Pillow on `ICN-C0419-S1000D0379-001-01.CGM` | `python3 -c "from PIL import Image; import sys; im=Image.open(sys.argv[1]); im.save(sys.argv[2])" "/home/hskim/projects/S1000D-RAG/docs/S1000D Issue 6/Bike Data Set for Release number 6 R2/ICN-C0419-S1000D0379-001-01.CGM" "/tmp/jarvis-runtime/s1000d-cgm-preview-test/ICN-C0419-S1000D0379-001-01.pillow.png"` | 1 | empty | `PIL.UnidentifiedImageError: cannot identify image file ... ICN-C0419-S1000D0379-001-01.CGM` | No |
| Pillow on `ICN-C0419-S1000D0380-001-01.CGM` | `python3 -c "from PIL import Image; import sys; im=Image.open(sys.argv[1]); im.save(sys.argv[2])" "/home/hskim/projects/S1000D-RAG/docs/S1000D Issue 6/Bike Data Set for Release number 6 R2/ICN-C0419-S1000D0380-001-01.CGM" "/tmp/jarvis-runtime/s1000d-cgm-preview-test/ICN-C0419-S1000D0380-001-01.pillow.png"` | 1 | empty | `PIL.UnidentifiedImageError: cannot identify image file ... ICN-C0419-S1000D0380-001-01.CGM` | No |
| Pillow on `ICN-C0419-S1000D0382-001-01.CGM` | `python3 -c "from PIL import Image; import sys; im=Image.open(sys.argv[1]); im.save(sys.argv[2])" "/home/hskim/projects/S1000D-RAG/docs/S1000D Issue 6/Bike Data Set for Release number 6 R2/ICN-C0419-S1000D0382-001-01.CGM" "/tmp/jarvis-runtime/s1000d-cgm-preview-test/ICN-C0419-S1000D0382-001-01.pillow.png"` | 1 | empty | `PIL.UnidentifiedImageError: cannot identify image file ... ICN-C0419-S1000D0382-001-01.CGM` | No |

## Produced files

No PNG files were produced by the conversion attempts.

Temporary investigation logs/scripts were written only under `/tmp/jarvis-runtime/s1000d-cgm-preview-test`:

- `tool_check.log`
- `sample_assets.log`
- `attempt_conversions.sh`
- `conversion_attempts.log`
- per-attempt stdout/stderr files

## Feasibility recommendation

Do not implement CGM preview generation as an enabled local feature in this environment unless a real CGM-capable converter is added or otherwise made available. The current environment can detect CGM files with `file`, but it cannot rasterize them.

Recommended application behavior for now:
1. Treat `.CGM` assets as known-but-unpreviewable.
2. Display a clear placeholder or download/open-original action instead of attempting preview conversion.
3. Gate any future CGM conversion behind an explicit converter availability check.

## Blockers

- No CGM-capable command-line converter is installed in `PATH`.
- Pillow does not support these binary CGM files.
- No PNG output exists to validate with `file`, `identify`, image dimensions, or visual inspection.

## Installed-tool retest on 2026-06-02

After installation was approved, the following Ubuntu packages were installed and tested:

- `imagemagick` 6.9.12-98
- `graphicsmagick` 1.3.42
- `libreoffice-draw` 24.2.7.2
- `inkscape` 1.2.2
- `default-jre-headless` / `libreoffice-java-common` to remove the LibreOffice Java warning

Retest output directory: `/tmp/jarvis-runtime/s1000d-cgm-preview-install-test`.

| Tool | Exact approach | Result |
|---|---|---|
| ImageMagick `convert` | `convert <asset>.CGM <asset>.im.png` | Failed: `no decode delegate for this image format 'CGM'`; no PNG produced. |
| GraphicsMagick `gm convert` | `gm convert <asset>.CGM <asset>.gm.png` | Failed because GraphicsMagick delegates CGM rendering to external `ralcgm`: `Delegate failed ("ralcgm" -d ps < "%i" > "%o" ...)`; no PNG produced. |
| LibreOffice Draw | `libreoffice --headless --convert-to png --outdir ... <asset>.CGM` | Failed: `Error: source file could not be loaded`; no PNG produced. Retesting after Java installation removed the Java warning but did not fix CGM import. |
| Inkscape | `inkscape <asset>.CGM --export-type=png --export-filename=...` | Failed: Inkscape could not detect the file format and tried/fail-opened it as SVG; no PNG produced. |
| Python packages `pycgm` / `cgm` | temporary venv under `/tmp/jarvis-runtime/cgm-python-venv` | Not applicable for these S1000D CGM images: `cgm` is a causal graphical model package; `pycgm` installs gait-analysis modules, not a Computer Graphics Metafile renderer. |

Additional finding: Ubuntu Noble does not provide an obvious `ralcgm`, `cgm2png`, `cgm2svg`, `uniconvertor`, or `sk1` package in the configured apt repositories. GraphicsMagick documentation explicitly says CGM support requires `ralcgm`, but the historical AGOCG URL now returns 404.

Updated conclusion: even after installing the common available graphics/conversion tools, these S1000D binary CGM v4 assets still cannot be converted to PNG in this WSL environment. A future preview implementation should either obtain a working `ralcgm`/dedicated CGM renderer from a trusted source, or treat CGM as unpreviewable and offer a placeholder/open-original behavior.

## Next implementation suggestion

Implement image evidence display with converter probing rather than assuming CGM preview support:

- show thumbnails only for browser-renderable or successfully converted assets,
- expose `.CGM` source files via a safe download/open-original endpoint,
- show a clear `CGM preview unavailable` placeholder when no renderer is available,
- optionally add a future `ralcgm`/dedicated-renderer probe that records converter command path, version, test conversion exit code, output `file` result, PNG size/dimensions, and fallback reason.
