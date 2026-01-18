export interface SampleFiles {
    audio: {
        fake: string[];
        real: string[];
    };
    image: {
        present: string[];
        absent: string[];
    };
}

export type SampleCategory = "audio/fake" | "audio/real" | "image/present" | "image/absent";

export async function fetchSampleFiles(): Promise<SampleFiles> {
    const response = await fetch("/api/samples");
    return response.json();
}

