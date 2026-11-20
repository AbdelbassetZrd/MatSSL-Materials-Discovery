import { Component } from "react";

/**
 * Keeps one failing widget (typically the WASM chemical editor) from taking the whole page
 * down, and shows what went wrong instead of a blank screen.
 */
export default class Boundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // eslint-disable-next-line no-console
    console.error("widget failed:", error, info);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900">
        <b>{this.props.title || "This widget could not start"}.</b>{" "}
        {this.props.hint || "You can keep using the rest of the page."}
        <pre className="mt-2 whitespace-pre-wrap break-all font-mono text-[11px] opacity-80">
          {String(this.state.error?.message || this.state.error)}
        </pre>
      </div>
    );
  }
}
