"use client";

import { useCallback, useState, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, X, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import SampleFileSelector from "@/components/SampleFileSelector";
import { SampleCategory, SampleFiles, fetchSampleFiles } from "@/lib/samples";

type FileType = "audio" | "image" | null;

interface FileDropzoneProps {
    onFileChange: (file: File | null, fileType: FileType) => void;
    onPredict?: () => void;
    canPredict?: boolean;
    isLoading?: boolean;
}

export default function FileDropzone({ onFileChange, onPredict, canPredict = false, isLoading = false }: FileDropzoneProps) {
    const [file, setFile] = useState<File | null>(null);
    const [fileType, setFileType] = useState<FileType>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [samplePath, setSamplePath] = useState<string | null>(null);
    const [samples, setSamples] = useState<SampleFiles | null>(null);

    // Précharger les samples dès le montage
    useEffect(() => {
        fetchSampleFiles().then(setSamples);
    }, []);

    const onDrop = useCallback((acceptedFiles: File[]) => {
        if (acceptedFiles.length === 0) return;

        const uploadedFile = acceptedFiles[0];
        setFile(uploadedFile);

        let detectedType: FileType = null;
        if (uploadedFile.type.startsWith("audio/")) {
            detectedType = "audio";
        } else if (uploadedFile.type.startsWith("image/")) {
            detectedType = "image";
        }
        setFileType(detectedType);

        const url = URL.createObjectURL(uploadedFile);
        setPreview(url);
        setSamplePath(null);

        onFileChange(uploadedFile, detectedType);
    }, [onFileChange]);

    const handleSampleSelect = async (category: SampleCategory, filename: string) => {
        const [type, subtype] = category.split("/");
        const path = `/${type}/${subtype}/${filename}`;
        setSamplePath(path);

        try {
            const response = await fetch(path);
            const blob = await response.blob();

            // Determine file type and MIME type
            const detectedType: FileType = type === "audio" ? "audio" : "image";
            const mimeType = type === "audio"
                ? "audio/wav"
                : `image/${filename.split(".").pop()}`;

            const sampleFile = new File([blob], filename, { type: mimeType });

            setFile(sampleFile);
            setFileType(detectedType);

            // Force new URL with timestamp to avoid caching
            const newUrl = `${path}?t=${Date.now()}`;
            setPreview(newUrl);

            onFileChange(sampleFile, detectedType);
        } catch (error) {
            console.error("Error loading sample file:", error);
        }
    };

    const removeFile = () => {
        if (preview && !samplePath) URL.revokeObjectURL(preview);
        setFile(null);
        setFileType(null);
        setPreview(null);
        setSamplePath(null);
        onFileChange(null, null);
    };

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            "audio/wav": [".wav"],
            "image/*": [".png", ".jpg", ".jpeg"],
        },
        maxFiles: 1,
    });

    return (
        <div className="w-full max-w-2xl">
            <Tabs defaultValue="upload" className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="upload">Upload File</TabsTrigger>
                    <TabsTrigger value="sample">Use Sample</TabsTrigger>
                </TabsList>

                <TabsContent value="upload" className="mt-4">
                    <div
                        {...getRootProps()}
                        className={`
                            border-2 border-dashed rounded-lg p-12 text-center cursor-pointer
                            transition-colors duration-200
                            ${isDragActive
                                ? "border-primary bg-primary/5"
                                : "border-gray-300 hover:border-primary/50"
                            }
                        `}
                    >
                        <input {...getInputProps()} />
                        <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                        {isDragActive ? (
                            <p className="text-lg">Drop the file here...</p>
                        ) : (
                            <div>
                                <p className="text-lg mb-2">
                                    Drag and drop a file here, or click to select
                                </p>
                                <p className="text-sm text-gray-500">
                                    Accepted formats: WAV, PNG, JPG, JPEG
                                </p>
                            </div>
                        )}
                    </div>
                </TabsContent>

                <TabsContent value="sample" className="mt-4">
                    <div className="border rounded-lg p-4 max-h-50 overflow-y-auto">
                        <SampleFileSelector
                            onSelect={handleSampleSelect}
                            selectedPath={samplePath}
                            samples={samples}
                        />
                    </div>
                </TabsContent>
            </Tabs>

            {/* Contenu en dessous - bouton et preview */}
            {file && preview && (
                <div className="space-y-6 mt-6">
                    {onPredict && (
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
                                <span className="flex items-center gap-2">
                                    <Play className="w-4 h-4" />
                                    Run Classification & XAI
                                </span>
                            )}
                        </Button>
                    )}

                    <div className="p-4 border rounded-lg relative">
                        <button
                            onClick={removeFile}
                            className="absolute top-2 right-2 p-1 hover:bg-gray-100 rounded"
                        >
                            <X className="w-5 h-5" />
                        </button>

                        <div className="space-y-2">
                            <p className="text-sm text-gray-600">
                                File: {file.name} ({fileType})
                            </p>
                            {samplePath && (
                                <p className="text-xs text-muted-foreground">
                                    Path: {samplePath}
                                </p>
                            )}
                        </div>

                        {fileType === "audio" && (
                            <audio key={preview} controls className="w-full mt-3">
                                <source src={preview} type={file.type} />
                                Your browser does not support the audio element.
                            </audio>
                        )}

                        {fileType === "image" && (
                            <img
                                src={preview}
                                alt="Preview"
                                className="max-w-full h-auto rounded-lg mt-3"
                            />
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
