import { createBrowserRouter } from "react-router";
import { LaunchScreen } from "./components/LaunchScreen";
import { LiveSession } from "./components/LiveSession";
import { SessionSummary } from "./components/SessionSummary";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: LaunchScreen,
  },
  {
    path: "/session",
    Component: LiveSession,
  },
  {
    path: "/summary",
    Component: SessionSummary,
  },
]);
