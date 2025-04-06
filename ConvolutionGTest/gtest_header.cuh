// --- ANSI Color Codes ---
#ifdef _WIN32 // Basic check for Windows console (might need more robust detection)
    // Windows console needs specific API calls or might support ANSI via WT, etc.
    // For simplicity, disable color on basic Windows detection.
const char* const ANSI_RED = "";
const char* const ANSI_GREEN = "";
const char* const ANSI_YELLOW = "";
const char* const ANSI_RESET = "";
#else
    // ANSI escape codes for colors (common on Linux/macOS)
const char* const ANSI_RED = "\033[1;31m"; // Bold Red
const char* const ANSI_GREEN = "\033[1;32m"; // Bold Green
const char* const ANSI_YELLOW = "\033[1;33m"; // Bold Yellow
const char* const ANSI_RESET = "\033[0m";    // Reset color
#endif

