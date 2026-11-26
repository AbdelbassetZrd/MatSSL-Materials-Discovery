import React from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { HashRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import Layout from "./components/Layout.jsx";
import { CompositionProvider } from "./lib/composition.jsx";
import Home from "./pages/Home.jsx";
import Studio from "./pages/Studio.jsx";
import Results from "./pages/Results.jsx";
import ResearchLab from "./pages/ResearchLab.jsx";
import Compare from "./pages/Compare.jsx";
import DesignLab from "./pages/DesignLab.jsx";
import MoleculeStudio from "./pages/MoleculeStudio.jsx";
import MoleculeResults from "./pages/MoleculeResults.jsx";
import Diseases from "./pages/Diseases.jsx";
import Screen from "./pages/Screen.jsx";
import { MoleculeProvider } from "./lib/molecule.jsx";

const qc = new QueryClient({ defaultOptions: { queries: { staleTime: 5 * 60 * 1000, retry: 1 } } });

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <QueryClientProvider client={qc}>
      <CompositionProvider>
        <MoleculeProvider>
        <HashRouter>
          <Routes>
            <Route element={<Layout />}>
              <Route index element={<Home />} />
              <Route path="studio" element={<Studio />} />
              <Route path="results" element={<Results />} />
              <Route path="lab" element={<ResearchLab />} />
              <Route path="compare" element={<Compare />} />
              <Route path="design" element={<DesignLab />} />
              <Route path="molecules" element={<MoleculeStudio />} />
              <Route path="molecule-results" element={<MoleculeResults />} />
              <Route path="diseases" element={<Diseases />} />
              <Route path="screen" element={<Screen />} />
            </Route>
          </Routes>
        </HashRouter>
        </MoleculeProvider>
      </CompositionProvider>
    </QueryClientProvider>
  </React.StrictMode>
);
