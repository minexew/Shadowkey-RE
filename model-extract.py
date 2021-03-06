#!/usr/bin/env python

import argparse
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import trimesh
import trimesh.visual


IMAGE_FRAMES = 20
ANIM_SECONDS = 3


parser = argparse.ArgumentParser()
parser.add_argument("models_huge", type=Path)
parser.add_argument("file_offset", type=int)
parser.add_argument("file_length", type=int)
parser.add_argument("output", type=Path)        # can be .bin or .glb or .gif
parser.add_argument("--image-size", type=int, default=500)
parser.add_argument("--save-texture", type=Path, default=[], action="append")   # can be .gif or .png
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()


offset, length = args.file_offset, args.file_length
assert length > 0


with open(args.models_huge, "rb") as f:
    if args.verbose:
        def dprint(*args, **kwargs):
            print(*args, **kwargs)
    else:
        def dprint(*args, **kwargs):
            pass

    f.seek(offset)
    modelbytes = f.read(length)

    if args.output.suffix.lower() == ".bin":
        args.output.write_bytes(modelbytes)

    unk0, num_frames, num_verts, num_UVs, num_faces, num_coords, unk6 = struct.unpack("<7H", modelbytes[:14])

    dprint(f"[{args.output}] unk0={unk0} num_frames={num_frames} num_verts={num_verts} num_UVs={num_UVs} num_faces={num_faces} num_coords={num_coords}")

    assert num_verts * 3 == num_coords
    # assert someheader[3] == someheader[4] * 3
    # num_following += 2

    pos = 14
    # words = struct.unpack(f"<{num_coords}h", modelbytes[pos:pos + num_coords * 2])
    coords = [struct.unpack("<hhh", modelbytes[pos + i * 6:pos + i * 6 + 6]) for i in range(num_coords // 3)]
    pos += num_coords * 2
    dprint("COORDS END AT", pos, coords)

    if num_frames > 1:
        count = (num_frames - 1) * num_coords // 3
        coords = [struct.unpack("<hhh", modelbytes[pos + i * 6:pos + i * 6 + 6]) for i in range(count)]
        pos += count * 6
        dprint("EXTRA FRAMES END AT", pos, coords)

    uvs = [tuple(a / 0x100 for a in struct.unpack("<HH", modelbytes[pos + i * 4:pos + i * 4 + 4])) for i in range(num_UVs)]
    pos += num_UVs * 4
    dprint("UVs END AT", pos, uvs)

    assert all(0 <= u <= 256.0 and 0 <= v <= 256.0 for u, v in uvs)

    faces = [struct.unpack("<HHHHHH", modelbytes[pos + i * 12:pos + i * 12 + 12]) for i in range(num_faces)]
    pos += num_faces * 12
    dprint("FACES END AT", pos, faces)

    num_tex_frames, = struct.unpack("<H", modelbytes[pos:pos + 2])
    dprint("NUM-TEXTURES", num_tex_frames)
    pos += 2

    w, h = struct.unpack("<HH", modelbytes[pos:pos+4])
    pos += 4
    dprint("IMAGE", w, h)

    assert w <= 128
    assert h <= 128

    tex_frames = []

    for frame_index in range(num_tex_frames):
        rgb565_words = np.frombuffer(modelbytes[pos:pos+w*h*2], dtype="uint16")
        pos += w*h*2

        # 0xf0f (violet) = transparent
        rgb = np.array([((rgb565 >> 8) * 0x11, ((rgb565 & 0x0f0) >> 4) * 0x11, (rgb565 & 0x00f) * 0x11, 0 if rgb565 == 0xf0f else 255) for rgb565 in rgb565_words], dtype=np.uint8).reshape((h, w, 4))
        # rgb = np.flip(rgb, axis=0).copy()  # the copy is necessary to make array C-contiguous

        img = Image.frombytes("RGBA", (rgb.shape[1], rgb.shape[0]), rgb)
        # img.save(f"DUMPMODEL{i}.png")

        tex_frames.append(img)

    if len(tex_frames):
        for path in args.save_texture:
            if path.suffix.lower() == ".gif":
                tex_frames[0].save(path, append_images=tex_frames[1:], save_all=True, duration=1000, loop=0)
            else:
                tex_frames[0].save(path)

    dprint("remaining", len(modelbytes[pos:]), "bytes; preview:", modelbytes[pos:pos + 16])
    # dprint(modelbytes[pos:])
    dprint()

    if len(tex_frames):
        # remap UVs
        uvs = [(u / img.width, 1 - v / img.height) for u, v in uvs]

    trimesh_verts = []
    trimesh_uvs = []
    for a, b, c, p, q, r in faces:
        trimesh_verts.append(coords[a])
        trimesh_verts.append(coords[b])
        trimesh_verts.append(coords[c])
        trimesh_uvs.append(uvs[p])
        trimesh_uvs.append(uvs[q])
        trimesh_uvs.append(uvs[r])

    mesh = trimesh.Trimesh(vertices=trimesh_verts, faces=[(3*i, 3*i+1, 3*i+2) for i in range(len(faces))], process=False, maintain_order=True)
    if len(tex_frames): mesh.visual = trimesh.visual.texture.TextureVisuals(uv=trimesh_uvs, image=tex_frames[0])
    mesh.apply_scale(0.01)

    if args.output.suffix.lower() in {".glb", ".obj"}:
        mesh.export(args.output)


if args.output.suffix.lower() == ".gif":
    import render_me_harder

    imgs = render_me_harder.render_frames(mesh, w=args.image_size, h=args.image_size, num_frames=IMAGE_FRAMES)

    # Save GIF image
    # https://stackoverflow.com/a/57751793
    LOOP_FOREVER = 0
    imgs[0].save(args.output, format="GIF", append_images=imgs[1:],
                 save_all=True, duration=ANIM_SECONDS / IMAGE_FRAMES * 1000, loop=LOOP_FOREVER)
