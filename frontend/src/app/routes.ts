import { createBrowserRouter } from "react-router";
import { LaunchScreen } from "./components/LaunchScreen";
import { SessionSummary } from "./components/SessionSummary";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: LaunchScreen,
  },
  {
    path: "/summary",
    Component: SessionSummary,
  },
]);
