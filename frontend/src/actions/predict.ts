"use server";

interface PredictRequest {
    model: string;
    xai_methods: string[];
    file_type: string;
    file_b64: string;
}

export async function predictWithXAI(request: PredictRequest) {
    const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/v1/classification_models/predict/`,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(request),
            cache: "no-store",
        }
    );

    if (!response.ok) {
        throw new Error("Failed to predict");
    }

    return response.json();
}
