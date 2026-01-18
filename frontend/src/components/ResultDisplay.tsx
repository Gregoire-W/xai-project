"use client";

import { useState } from "react";
import { Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";

interface ResultDisplayProps {
    image: string;
    prediction: string[];
    xaiResults: Record<string, string>;
    onSave: (name: string) => void;
}

export default function ResultDisplay({
    image,
    prediction,
    xaiResults,
    onSave,
}: ResultDisplayProps) {
    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const [resultName, setResultName] = useState("");

    const handleSave = () => {
        if (resultName.trim()) {
            onSave(resultName.trim());
            setIsDialogOpen(false);
            setResultName("");
        }
    };

    return (
        <div className="w-full max-w-4xl space-y-6">
            {/* Header with save button */}
            <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold">Classification Result</h2>
                <Button onClick={() => setIsDialogOpen(true)} variant="default" size="sm">
                    <Save className="w-4 h-4 mr-2" />
                    Save
                </Button>
            </div>

            {/* Prediction Badge */}
            <div className="flex items-center gap-3">
                <span className="text-sm font-medium text-gray-600">Prediction:</span>
                <div className="flex gap-2 flex-wrap">
                    {prediction.map((label, index) => (
                        <Badge key={index} className="text-md px-4" variant="default">
                            {label}
                        </Badge>
                    ))}
                </div>
            </div>

            {/* Image Display */}
            <div className="border rounded-lg p-4 bg-white shadow-sm flex justify-center">
                <img
                    src={`data:image/png;base64,${image}`}
                    alt="Analysis result"
                    className="max-w-full h-auto rounded-lg"
                    style={{ objectFit: "contain" }}
                />
            </div>

            {/* XAI Results */}
            {Object.keys(xaiResults).length > 0 && (
                <div className="border rounded-lg p-6 bg-white shadow-sm space-y-4">
                    <h3 className="text-xl font-semibold">XAI Methods</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {Object.entries(xaiResults).map(([method, imageB64]) => (
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

            {/* Save Dialog */}
            <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Save Result</DialogTitle>
                    </DialogHeader>
                    <div className="py-4">
                        <Input
                            placeholder="Enter a name for this result"
                            value={resultName}
                            onChange={(e) => setResultName(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && handleSave()}
                        />
                    </div>
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setIsDialogOpen(false)}>
                            Cancel
                        </Button>
                        <Button onClick={handleSave} disabled={!resultName.trim()}>
                            Save
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div>
    );
}
