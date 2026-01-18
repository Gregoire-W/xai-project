"use client";

import {
    Sidebar,
    SidebarContent,
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarMenu,
    SidebarMenuItem,
} from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { Trash2, History, FileText, Plus } from "lucide-react";
import { SavedResult } from "@/hooks/useSavedResults";

interface AppSidebarProps {
    savedResults: SavedResult[];
    onResultClick: (result: SavedResult) => void;
    onDelete: (id: string) => void;
    onNewAnalysis: () => void;
}

export default function AppSidebar({ savedResults, onResultClick, onDelete, onNewAnalysis }: AppSidebarProps) {
    return (
        <Sidebar>
            <SidebarContent>
                {/* New Analysis Button */}
                <div className="p-4">
                    <Button
                        onClick={onNewAnalysis}
                        className="w-full"
                        size="sm"
                    >
                        <Plus className="h-4 w-4 mr-2" />
                        New Analysis
                    </Button>
                </div>

                <SidebarGroup>
                    <SidebarGroupLabel className="flex items-center gap-2 text-base font-semibold">
                        <History className="h-5 w-5" />
                        <span>Saved Results</span>
                        <span className="ml-auto text-xs font-normal text-muted-foreground">
                            {savedResults.length}
                        </span>
                    </SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            {savedResults.length === 0 ? (
                                <div className="px-2 py-8 text-center">
                                    <p className="text-sm text-muted-foreground">No saved results yet</p>
                                </div>
                            ) : (
                                savedResults.map((result) => (
                                    <SidebarMenuItem key={result.id}>
                                        <div
                                            className="group flex items-start justify-between gap-2 w-full p-2 rounded-md hover:bg-accent cursor-pointer transition-colors"
                                            onClick={() => onResultClick(result)}
                                        >
                                            <div className="flex items-start gap-2 flex-1 min-w-0">
                                                <FileText className="h-4 w-4 mt-0.5 text-muted-foreground flex-shrink-0" />
                                                <div className="flex-1 min-w-0">
                                                    <p className="text-sm font-medium truncate">{result.name}</p>
                                                    <p className="text-xs text-muted-foreground">
                                                        {new Date(result.timestamp).toLocaleDateString()}
                                                    </p>
                                                </div>
                                            </div>
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    onDelete(result.id);
                                                }}
                                            >
                                                <Trash2 className="h-4 w-4 text-destructive" />
                                            </Button>
                                        </div>
                                    </SidebarMenuItem>
                                ))
                            )}
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>
            </SidebarContent>
        </Sidebar>
    );
}
