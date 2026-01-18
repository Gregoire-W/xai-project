"use client";

import { useEffect, useState } from "react";
import FileDropzone from "@/components/FileDropzone";
import SelectionPanel from "@/components/SelectionPanel";
import ResultDisplay from "@/components/ResultDisplay";
import SavedResultDetail from "@/components/SavedResultDetail";
import AppSidebar from "@/components/Sidebar";
import { SidebarProvider } from "@/components/ui/sidebar";
import { getModelList } from "@/actions/models";
import { predictWithXAI } from "@/actions/predict";
import { useSavedResults, SavedResult } from "@/hooks/useSavedResults";

type FileType = "audio" | "image" | null;

interface ModelList {
  audio_model: string[];
  image_model: string[];
}

interface PredictionResult {
  image: string;
  prediction: string[];
  xai_results: Record<string, string>;
}

export default function Home() {
  const [models, setModels] = useState<ModelList | null>(null);
  const [fileType, setFileType] = useState<FileType>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [selectedMethods, setSelectedMethods] = useState<string[]>([]);
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [usedMethods, setUsedMethods] = useState<string[]>([]);
  const [viewingSavedResult, setViewingSavedResult] = useState<SavedResult | null>(null);
  const { savedResults, saveResult, deleteResult } = useSavedResults();

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const data = await getModelList();
        setModels(data);
      } catch (error) {
        console.error("Error fetching models:", error);
      }
    };

    fetchModels();
  }, []);

  const handleFileChange = (file: File | null, type: FileType) => {
    setFileType(type);
    setCurrentFile(file);
    setSelectedModel(null);
    setSelectedMethods([]);
    setUsedMethods([]);
    setResult(null);
  };

  const handlePredict = async () => {
    if (!currentFile || !selectedModel || !fileType || selectedMethods.length === 0) return;

    // Extract model name from "type:modelname" format
    const modelName = selectedModel.split(":")[1] || selectedModel;

    setIsLoading(true);
    try {
      const reader = new FileReader();

      const base64Promise = new Promise<string>((resolve, reject) => {
        reader.onloadend = () => {
          const base64 = reader.result?.toString().split(",")[1] || "";
          resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(currentFile);
      });

      const base64 = await base64Promise;

      const response = await predictWithXAI({
        model: modelName,
        xai_methods: selectedMethods,
        file_type: fileType,
        file_b64: base64,
      });

      if (response.success) {
        console.log("Success analysis: ", response);
        setResult((prevResult) => ({
          image: response.data.image,
          prediction: response.data.prediction || [],
          xai_results: {
            ...(prevResult?.xai_results || {}),
            ...response.data.xai_results,
          },
        }));
        setUsedMethods((prev) => [...new Set([...prev, ...selectedMethods])]);
        setSelectedMethods([]);
      } else {
        console.error("Prediction failed:", response.data.error);
        handleReset();
      }
    } catch (error) {
      console.error("Prediction error:", error);
      handleReset();
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setCurrentFile(null);
    setFileType(null);
    setSelectedModel(null);
    setSelectedMethods([]);
    setUsedMethods([]);
    setViewingSavedResult(null);
  };

  const handleSaveResult = (name: string) => {
    if (result && selectedModel) {
      const modelName = selectedModel.split(":")[1] || selectedModel;
      saveResult(name, modelName, result);
    }
  };

  const handleLoadResult = (savedResult: SavedResult) => {
    setViewingSavedResult(savedResult);
  };

  const canPredict = !!currentFile && !!selectedModel && selectedMethods.length > 0;

  const allModels = [
    ...(models?.audio_model || []),
    ...(models?.image_model || []),
  ];

  const availableModels = fileType
    ? fileType === "audio"
      ? models?.audio_model || []
      : models?.image_model || []
    : allModels;

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full">
        {/* Sidebar */}
        <AppSidebar
          savedResults={savedResults}
          onResultClick={handleLoadResult}
          onDelete={deleteResult}
          onNewAnalysis={handleReset}
        />

        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <header className="bg-primary text-white py-6 px-8 shadow-lg">
            <div className="max-w-7xl mx-auto">
              <h1 className="text-3xl font-bold tracking-tight">Explainability AI</h1>
              <p className="text-blue-100 mt-1">Deep Learning Model Classification with XAI Methods</p>
            </div>
          </header>

          {/* Content Area */}
          <div className="flex-1 p-8">
            <div className="max-w-7xl mx-auto flex gap-8 justify-center items-start">
              {viewingSavedResult ? (
                <SavedResultDetail
                  result={viewingSavedResult}
                  onClose={() => setViewingSavedResult(null)}
                />
              ) : (
                <>
                  <SelectionPanel
                    models={models}
                    fileType={fileType}
                    selectedModel={selectedModel}
                    onModelChange={setSelectedModel}
                    selectedMethods={selectedMethods}
                    onMethodsChange={setSelectedMethods}
                    disableModelSelection={!!result}
                    usedMethods={usedMethods}
                    onPredict={handlePredict}
                    canPredict={canPredict}
                    isLoading={isLoading}
                  />
                  {result ? (
                    <ResultDisplay
                      image={result.image}
                      prediction={result.prediction}
                      xaiResults={result.xai_results}
                      onSave={handleSaveResult}
                    />
                  ) : (
                    <FileDropzone
                      onFileChange={handleFileChange}
                      onPredict={handlePredict}
                      canPredict={canPredict}
                      isLoading={isLoading}
                    />
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </SidebarProvider>
  );
}
