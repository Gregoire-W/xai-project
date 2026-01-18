"use client";

import { useState, useEffect } from "react";

export interface SavedResult {
    id: string;
    name: string;
    timestamp: number;
    model: string;
    image: string;
    prediction: string[];
    xai_results: Record<string, string>;
}

const STORAGE_KEY = "xai_saved_results";

export function useSavedResults() {
    const [savedResults, setSavedResults] = useState<SavedResult[]>([]);

    useEffect(() => {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
            setSavedResults(JSON.parse(stored));
        }
    }, []);

    const saveResult = (
        name: string,
        model: string,
        result: { image: string; prediction: string[]; xai_results: Record<string, string> }
    ) => {
        const newResult: SavedResult = {
            id: Date.now().toString(),
            name,
            timestamp: Date.now(),
            model,
            ...result,
        };

        const updated = [...savedResults, newResult];
        setSavedResults(updated);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    };

    const deleteResult = (id: string) => {
        const updated = savedResults.filter((r) => r.id !== id);
        setSavedResults(updated);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    };

    return { savedResults, saveResult, deleteResult };
}
