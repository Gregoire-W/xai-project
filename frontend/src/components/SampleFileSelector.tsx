"use client";

import { Folder, FileAudio, FileImage } from "lucide-react";
import { SampleCategory, SampleFiles } from "@/lib/samples";

interface SampleFileSelectorProps {
    onSelect: (category: SampleCategory, filename: string) => void;
    selectedPath: string | null;
    samples: SampleFiles | null;
}

export default function SampleFileSelector({ onSelect, selectedPath, samples }: SampleFileSelectorProps) {

    if (!samples) {
        return <div className="text-sm text-muted-foreground">Loading samples...</div>;
    }

    return (
        <div className="space-y-4">
            {/* Audio Samples */}
            <div>
                <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                    <Folder className="h-4 w-4" />
                    Audio Samples
                </h3>

                <div className="space-y-3">
                    {/* Fake Audio */}
                    <div>
                        <p className="text-xs text-muted-foreground mb-2">audio/fake/</p>
                        <div className="space-y-1">
                            {samples.audio.fake.map((file) => {
                                const fullPath = `/audio/fake/${file}`;
                                const isSelected = selectedPath === fullPath;
                                return (
                                    <button
                                        key={file}
                                        onClick={() => onSelect("audio/fake", file)}
                                        className={`w-full text-left px-3 py-2 rounded text-sm flex items-center gap-2 transition-colors ${isSelected
                                            ? "bg-primary text-primary-foreground"
                                            : "hover:bg-accent"
                                            }`}
                                    >
                                        <FileAudio className="h-4 w-4 shrink-0" />
                                        <span className="truncate">{file}</span>
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* Real Audio */}
                    <div>
                        <p className="text-xs text-muted-foreground mb-2">audio/real/</p>
                        <div className="space-y-1">
                            {samples.audio.real.map((file) => {
                                const fullPath = `/audio/real/${file}`;
                                const isSelected = selectedPath === fullPath;
                                return (
                                    <button
                                        key={file}
                                        onClick={() => onSelect("audio/real", file)}
                                        className={`w-full text-left px-3 py-2 rounded text-sm flex items-center gap-2 transition-colors ${isSelected
                                            ? "bg-primary text-primary-foreground"
                                            : "hover:bg-accent"
                                            }`}
                                    >
                                        <FileAudio className="h-4 w-4 shrink-0" />
                                        <span className="truncate">{file}</span>
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                </div>
            </div>

            {/* Image Samples (CheXpert) */}
            <div>
                <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                    <Folder className="h-4 w-4" />
                    Image Samples (CheXpert)
                </h3>

                <div className="space-y-3">
                    {/* Present */}
                    <div>
                        <p className="text-xs text-muted-foreground mb-2">image/present/</p>
                        <div className="space-y-1">
                            {samples.image.present.map((file) => {
                                const fullPath = `/image/present/${file}`;
                                const isSelected = selectedPath === fullPath;
                                return (
                                    <button
                                        key={file}
                                        onClick={() => onSelect("image/present", file)}
                                        className={`w-full text-left px-3 py-2 rounded text-sm flex items-center gap-2 transition-colors ${isSelected
                                            ? "bg-primary text-primary-foreground"
                                            : "hover:bg-accent"
                                            }`}
                                    >
                                        <FileImage className="h-4 w-4 shrink-0" />
                                        <span className="truncate">{file}</span>
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* Absent */}
                    <div>
                        <p className="text-xs text-muted-foreground mb-2">image/absent/</p>
                        <div className="space-y-1">
                            {samples.image.absent.map((file) => {
                                const fullPath = `/image/absent/${file}`;
                                const isSelected = selectedPath === fullPath;
                                return (
                                    <button
                                        key={file}
                                        onClick={() => onSelect("image/absent", file)}
                                        className={`w-full text-left px-3 py-2 rounded text-sm flex items-center gap-2 transition-colors ${isSelected
                                            ? "bg-primary text-primary-foreground"
                                            : "hover:bg-accent"
                                            }`}
                                    >
                                        <FileImage className="h-4 w-4 shrink-0" />
                                        <span className="truncate">{file}</span>
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
