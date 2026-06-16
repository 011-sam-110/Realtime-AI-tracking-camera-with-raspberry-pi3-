import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "@/components/Layout";
import Live from "@/views/Live";
import People from "@/views/People";
import Person from "@/views/Person";
import Events from "@/views/Events";
import Analytics from "@/views/Analytics";
import Settings from "@/views/Settings";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Live />} />
        <Route path="people" element={<People />} />
        <Route path="people/:id" element={<Person />} />
        <Route path="events" element={<Events />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
