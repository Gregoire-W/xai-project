"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

interface ModelList {
    audio_model: string[];
    image_model: string[];
}

type FileType = "audio" | "image" | null;

interface SelectionPanelProps {
    models: ModelList | null;
    fileType: FileType;
    selectedModel: string | null;
    onModelChange: (model: string) => void;
    selectedMethods: string[];
    onMethodsChange: (methods: string[]) => void;
    disableModelSelection?: boolean;
    usedMethods?: string[];
    onPredict?: () => void;
    canPredict?: boolean;
    isLoading?: boolean;
}

const XAI_METHODS = ["LIME", "Grad-CAM", "SHAP"];

export default function SelectionPanel({
    models,
    fileType,
    selectedModel,
    onModelChange,
    selectedMethods,
    onMethodsChange,
    disableModelSelection = false,
    usedMethods = [],
    onPredict,
    canPredict = false,
    isLoading = false,
}: SelectionPanelProps) {
    const toggleMethod = (method: string) => {
        if (!fileType || usedMethods.includes(method)) return;

        if (selectedMethods.includes(method)) {
            onMethodsChange(selectedMethods.filter((m) => m !== method));
        } else {
            onMethodsChange([...selectedMethods, method]);
        }
    };

    const isModelClickable = (modelType: "audio" | "image") => {
        return fileType === modelType && !disableModelSelection;
    };

    return (
        <div className="w-full max-w-md p-6 border rounded-lg space-y-6">
            {/* Model Selection */}
            <div>
                <h3 className="text-lg font-semibold mb-3">Models</h3>
                {!models ? (
                    <p className="text-sm text-gray-500">Loading models...</p>
                ) : (
                    <div className="space-y-4">
                        {/* Audio Models */}
                        {models.audio_model.length > 0 && (
                            <div>
                                <h4 className="text-sm font-semibold text-gray-600 mb-2">Audio</h4>
                                <div className="space-y-2">
                                    {models.audio_model.map((model) => {
                                        const isClickable = isModelClickable("audio");
                                        const modelKey = `audio:${model}`;
                                        const isSelected = selectedModel === modelKey;
                                        return (
                                            <button
                                                key={`audio-${model}`}
                                                onClick={() => isClickable && onModelChange(modelKey)}
                                                disabled={!isClickable}
                                                className={`
                                                    w-full px-4 py-3 rounded-lg border transition-all
                                                    flex items-center justify-between
                                                    ${!isClickable ? "cursor-not-allowed opacity-50" : "cursor-pointer hover:bg-gray-50"}
                                                    ${isSelected
                                                        ? "border-primary bg-primary/5"
                                                        : "border-gray-200"
                                                    }
                                                `}
                                            >
                                                <span className="text-sm font-medium">{model}</span>
                                                {isSelected && (
                                                    <Badge className="bg-primary">Active</Badge>
                                                )}
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>
                        )}

                        {/* Image Models */}
                        {models.image_model.length > 0 && (
                            <div>
                                <h4 className="text-sm font-semibold text-gray-600 mb-2">Image</h4>
                                <div className="space-y-2">
                                    {models.image_model.map((model) => {
                                        const isClickable = isModelClickable("image");
                                        const modelKey = `image:${model}`;
                                        const isSelected = selectedModel === modelKey;
                                        return (
                                            <button
                                                key={`image-${model}`}
                                                onClick={() => isClickable && onModelChange(modelKey)}
                                                disabled={!isClickable}
                                                className={`
                                                    w-full px-4 py-3 rounded-lg border transition-all
                                                    flex items-center justify-between
                                                    ${!isClickable ? "cursor-not-allowed opacity-50" : "cursor-pointer hover:bg-gray-50"}
                                                    ${isSelected
                                                        ? "border-primary bg-primary/5"
                                                        : "border-gray-200"
                                                    }
                                                `}
                                            >
                                                <span className="text-sm font-medium">{model}</span>
                                                {isSelected && (
                                                    <Badge className="bg-primary">Active</Badge>
                                                )}
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* XAI Methods Selection */}
            <div>
                <h3 className="text-lg font-semibold mb-3">XAI Methods</h3>
                <div className="space-y-2">
                    {XAI_METHODS.map((method) => {
                        const isUsed = usedMethods.includes(method);
                        const isDisabled = !fileType || isUsed;
                        return (
                            <button
                                key={method}
                                onClick={() => toggleMethod(method)}
                                disabled={isDisabled}
                                className={`
                                w-full px-4 py-3 rounded-lg border transition-all
                                flex items-center justify-between
                                ${isDisabled ? "cursor-not-allowed opacity-50" : "cursor-pointer hover:bg-gray-50"}
                                ${selectedMethods.includes(method)
                                        ? "border-primary bg-primary/5"
                                        : "border-gray-200"
                                    }
                            `}
                            >
                                <span className="text-sm font-medium">{method}</span>
                                {selectedMethods.includes(method) && (
                                    <Badge className="bg-primary">Active</Badge>
                                )}
                                {isUsed && !selectedMethods.includes(method) && (
                                    <Badge variant="secondary">Used</Badge>
                                )}
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* Run Analysis Button - Only show if methods have been used (continuing analysis) */}
            {onPredict && usedMethods.length > 0 && (
                <Button
                    onClick={onPredict}
                    disabled={!canPredict || isLoading}
                    className="w-full"
                    size="lg"
                >
                    {isLoading ? (
                        <span className="flex items-center gap-2">
                            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                            Processing...
                        </span>
                    ) : (
                        "Run Analysis"
                    )}
                </Button>
            )}
        </div>
    );
}
