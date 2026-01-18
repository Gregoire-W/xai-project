"use client";

import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { SavedResult } from "@/hooks/useSavedResults";

interface SavedResultDetailProps {
    result: SavedResult;
    onClose: () => void;
}

export default function SavedResultDetail({ result, onClose }: SavedResultDetailProps) {
    return (
        <div className="w-7/10 space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold">{result.name}</h2>
                    <div className="flex gap-4 mt-2 text-sm text-gray-600">
                        <span>Model: <span className="font-medium">{result.model}</span></span>
                        <span>Date: <span className="font-medium">{new Date(result.timestamp).toLocaleString()}</span></span>
                    </div>
                </div>
                <Button onClick={onClose} variant="outline" size="sm">
                    <X className="w-4 h-4 mr-2" />
                    Close
                </Button>
            </div>

            {/* Prediction */}
            <div className="flex items-center gap-3">
                <span className="text-sm font-medium text-gray-600">Prediction:</span>
                <div className="flex gap-2 flex-wrap">
                    {result.prediction.map((label, index) => (
                        <Badge key={index} className="text-md px-4" variant="default">
                            {label}
                        </Badge>
                    ))}
                </div>
            </div>

            {/* Image */}
            <div className="border rounded-lg p-4 bg-white shadow-sm flex justify-center">
                <img
                    src={`data:image/png;base64,${result.image}`}
                    alt="Analysis result"
                    className="max-w-full h-auto rounded-lg"
                    style={{ objectFit: "contain" }}
                />
            </div>

            {/* XAI Results */}
            {Object.keys(result.xai_results).length > 0 && (
                <div className="border rounded-lg p-6 bg-white shadow-sm space-y-4">
                    <h3 className="text-xl font-semibold">XAI Methods</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {Object.entries(result.xai_results).map(([method, imageB64]) => (
                            <div key={method} className="space-y-2">
                                <h4 className="text-sm font-medium text-gray-700">{method}</h4>
                                <div className="border rounded-lg p-2 bg-gray-50">
                                    <img
                                        src={`data:image/png;base64,${imageB64}`}
                                        alt={`${method} result`}
                                        className="w-full h-auto"
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
