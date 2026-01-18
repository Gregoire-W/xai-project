import { NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";

export async function GET() {
    try {
        const publicDir = path.join(process.cwd(), "public");

        const audioFakeDir = path.join(publicDir, "audio", "fake");
        const audioRealDir = path.join(publicDir, "audio", "real");
        const imagePresentDir = path.join(publicDir, "image", "present");
        const imageAbsentDir = path.join(publicDir, "image", "absent");

        const [audioFake, audioReal, imagePresent, imageAbsent] = await Promise.all([
            fs.readdir(audioFakeDir),
            fs.readdir(audioRealDir),
            fs.readdir(imagePresentDir),
            fs.readdir(imageAbsentDir),
        ]);

        const samples = {
            audio: {
                fake: audioFake.filter((f) => f.endsWith(".wav")),
                real: audioReal.filter((f) => f.endsWith(".wav")),
            },
            image: {
                present: imagePresent.filter((f) => /\.(png|jpg|jpeg)$/i.test(f)),
                absent: imageAbsent.filter((f) => /\.(png|jpg|jpeg)$/i.test(f)),
            },
        };

        return NextResponse.json(samples);
    } catch (error) {
        console.error("Error reading sample files:", error);
        return NextResponse.json({
            audio: { fake: [], real: [] },
            image: { present: [], absent: [] }
        }, { status: 500 });
    }
}
