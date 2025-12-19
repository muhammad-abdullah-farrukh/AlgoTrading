import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { SidebarProvider, SidebarTrigger, SidebarInset } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { TradesProvider } from "@/contexts/TradesContext";
import Dashboard from "./pages/Dashboard";
import MT5Connection from "./pages/MT5Connection";
import TradeHistory from "./pages/TradeHistory";
import Trading from "./pages/Trading";
import Strategies from "./pages/Strategies";
import Autotrading from "./pages/Autotrading";
import ModelDashboard from "./pages/ModelDashboard";
import WebScraper from "./pages/WebScraper";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <TradesProvider>
        <div className="dark">
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <SidebarProvider defaultOpen={true}>
              <div className="min-h-screen flex w-full bg-background">
                <AppSidebar />
                <SidebarInset className="flex-1">
                  <header className="flex h-14 items-center gap-4 border-b border-border px-4">
                    <SidebarTrigger className="-ml-1" />
                    <div className="flex-1" />
                  </header>
                  <main className="flex-1 p-6">
                    <Routes>
                      <Route path="/" element={<Dashboard />} />
                      <Route path="/connection" element={<MT5Connection />} />
                      <Route path="/history" element={<TradeHistory />} />
                      <Route path="/trading" element={<Trading />} />
                      <Route path="/strategies" element={<Strategies />} />
                      <Route path="/autotrading" element={<Autotrading />} />
                      <Route path="/models" element={<ModelDashboard />} />
                      <Route path="/scraper" element={<WebScraper />} />
                      <Route path="*" element={<NotFound />} />
                    </Routes>
                  </main>
                </SidebarInset>
              </div>
            </SidebarProvider>
          </BrowserRouter>
        </div>
      </TradesProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
