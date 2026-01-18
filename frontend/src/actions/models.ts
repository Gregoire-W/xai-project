"use server";

interface ModelList {
    audio_model: string[];
    image_model: string[];
}

export async function getModelList(): Promise<ModelList> {
    console.info("Try to fetch model list with url: ", process.env.NEXT_PUBLIC_BACKEND_URL);
    try {
        const response = await fetch(
            `${process.env.NEXT_PUBLIC_BACKEND_URL}/v1/classification_models/model_list/`,
            { cache: "no-store" }
        );
        if (!response.ok) {
            throw new Error("Failed to fetch models");
        }

        return response.json();
    } catch (error) {
        console.error("Error while fetching models: ", error)
        throw new Error("Failed to fetch models");
    }
}