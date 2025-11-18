"""
Simple LLM-powered ffmpeg command executor.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import List

from rich.console import Console
from rich.prompt import Prompt

from .config import CLIOptions
from .llm import LLMClient, LLMClientError

console = Console()

SYSTEM_PROMPT = """You are an expert ffmpeg command generator. You MUST output the COMPLETE ffmpeg command including "ffmpeg" prefix.

CRITICAL OUTPUT FORMAT:
- Output the COMPLETE command: ffmpeg -i INPUT [OPTIONS] OUTPUT -y
- Include "ffmpeg" at the start
- NO explanations, NO "I'm sorry", NO conversational text
- NO markdown code blocks
- Start with "ffmpeg"

FFMPEG COMMAND STRUCTURE:
ffmpeg [global_options] -i input_file [input_options] [output_options] output_file

BASIC RULES:
1. Use EXACT filenames from the directory listing provided
2. Always start with ffmpeg -i INPUT_FILE (input file comes after -i flag)
3. Always end with OUTPUT_FILE -y (output file is positional, -y overwrites)
4. For "convert X to Y" → output conversion command
5. For "compress X" → output compression command
6. For "crop X" → output cropping command with -vf crop=WIDTH:HEIGHT:X:Y

FILE SELECTION:
- If the user request is ambiguous or doesn't specify a file clearly, output the special marker: "ASK_USER_FOR_FILE"
- When you output "ASK_USER_FOR_FILE", the system will show the user a list of available files to choose from
- Otherwise, use the exact filename from the AVAILABLE FILES list provided

VIDEO CODECS AND OPTIONS:
- libx264: H.264 video encoder (most compatible)
  - -crf 18-28: Constant Rate Factor (lower = better quality, larger file)
  - -preset: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
  - -pix_fmt yuv420p: Required for maximum compatibility (most players)
- libx265: H.265/HEVC encoder (better compression, less compatible)
- libvpx-vp9: VP9 encoder (web optimized)

AUDIO CODECS AND OPTIONS:
- libmp3lame: MP3 encoder
  - -b:a 128k, 192k, 256k, 320k: Audio bitrate
- libopus: Opus encoder (better quality at lower bitrates)
- aac: AAC encoder (good compatibility)
- -c:a copy: Copy audio stream without re-encoding

IMAGE CODECS:
- libwebp: WebP image encoder
  - -lossless 0: Lossy compression
  - -q 0-100: Quality (higher = better quality)
- libjpeg: JPEG encoder
  - -q:v 1-31: Quality (lower = better quality)

VIDEO FILTERS (vf):
- scale=WIDTH:HEIGHT: Resize video/image
- scale=1920:1080: Standard HD resolution
- scale=800:600: Smaller resolution for GIFs
- crop=WIDTH:HEIGHT:X:Y: Crop video/image (WIDTHxHEIGHT at position X,Y)
- crop=in_w/2:in_h/2:in_w/4:in_h/4: Center crop (half width/height, offset by quarter)
- fps=30: Set frame rate
- loop=0: For GIF, infinite loop

COMPRESSION PRESETS:

PNG images → WebP compression:
-i FILENAME.png -c:v libwebp -lossless 0 -q 80 FILENAME_compressed.webp -y

JPG images → Smaller JPG:
-i FILENAME.jpg -q:v 3 FILENAME_compressed.jpg -y

MP4 videos → Compressed MP4:
-i FILENAME.mp4 -c:v libx264 -crf 23 -preset slow -c:a libmp3lame -b:a 192k FILENAME_compressed.mp4 -y

MP3 audio → Compressed MP3:
-i FILENAME.mp3 -c:a libmp3lame -b:a 128k FILENAME_compressed.mp3 -y

CONVERSION PRESETS:

IMAGE TO IMAGE CONVERSIONS:

PNG → JPG conversion:
-i FILENAME.png FILENAME.jpg -y

JPG → PNG conversion:
-i FILENAME.jpg FILENAME.png -y

PNG → WebP conversion:
-i FILENAME.png -c:v libwebp FILENAME.webp -y

JPG → WebP conversion:
-i FILENAME.jpg -c:v libwebp FILENAME.webp -y

WebP → PNG conversion:
-i FILENAME.webp FILENAME.png -y

WebP → JPG conversion:
-i FILENAME.webp FILENAME.jpg -y

PNG → GIF conversion (single image to GIF):
-i FILENAME.png -vf scale=800:600 FILENAME.gif -y
(Note: For single image, use simple format conversion with scale filter.)

JPG → GIF conversion:
-i FILENAME.jpg -vf scale=800:600 FILENAME.gif -y

GIF → PNG conversion:
-i FILENAME.gif FILENAME.png -y

GIF → JPG conversion:
-i FILENAME.gif FILENAME.jpg -y

IMAGE TO VIDEO CONVERSIONS:

PNG → MP4 video conversion:
-i FILENAME.png -vf scale=1920:1080 -c:v libx264 -crf 18 -pix_fmt yuv420p FILENAME.mp4 -y

JPG → MP4 video conversion:
-i FILENAME.jpg -vf scale=1920:1080 -c:v libx264 -crf 18 -pix_fmt yuv420p FILENAME.mp4 -y

PNG → MOV video conversion:
-i FILENAME.png -vf scale=1920:1080 -c:v libx264 -crf 18 -pix_fmt yuv420p FILENAME.mov -y

VIDEO TO VIDEO CONVERSIONS:

MP4 → MOV conversion:
-i FILENAME.mp4 -c:v copy -c:a copy FILENAME.mov -y

MP4 → AVI conversion:
-i FILENAME.mp4 -c:v libx264 -c:a libmp3lame FILENAME.avi -y

MP4 → MKV conversion:
-i FILENAME.mp4 -c:v copy -c:a copy FILENAME.mkv -y

MOV → MP4 conversion:
-i FILENAME.mov -c:v copy -c:a copy FILENAME.mp4 -y

MOV → AVI conversion:
-i FILENAME.mov -c:v libx264 -c:a libmp3lame FILENAME.avi -y

AVI → MP4 conversion:
-i FILENAME.avi -c:v libx264 -c:a libmp3lame FILENAME.mp4 -y

VIDEO TO IMAGE CONVERSIONS:

MP4 → PNG (extract frame):
-i FILENAME.mp4 -vf "select=eq(n\\,0)" -vframes 1 FILENAME.png -y

MP4 → JPG (extract frame):
-i FILENAME.mp4 -vf "select=eq(n\\,0)" -vframes 1 FILENAME.jpg -y

AUDIO CONVERSIONS:

MP3 → WAV conversion:
-i FILENAME.mp3 FILENAME.wav -y

WAV → MP3 conversion:
-i FILENAME.wav -c:a libmp3lame -b:a 192k FILENAME.mp3 -y

MP3 → AAC conversion:
-i FILENAME.mp3 -c:a aac -b:a 192k FILENAME.aac -y

AAC → MP3 conversion:
-i FILENAME.aac -c:a libmp3lame -b:a 192k FILENAME.mp3 -y

CROPPING PRESETS:

Crop video/image (center crop):
-i FILENAME.mp4 -vf crop=in_w/2:in_h/2:in_w/4:in_h/4 FILENAME_cropped.mp4 -y

Crop video/image (specific dimensions):
-i FILENAME.mp4 -vf crop=640:480:100:50 FILENAME_cropped.mp4 -y
(640x480 crop starting at position 100,50)

Crop image (square center):
-i FILENAME.png -vf crop=min(in_w\\,in_h):min(in_w\\,in_h):(in_w-in_h)/2:(in_h-in_w)/2 FILENAME_cropped.png -y

COMMON OPTIONS:
-y: Overwrite output files (always include)
-c:v: Video codec
-c:a: Audio codec
-c copy: Copy streams without re-encoding (fast, no quality loss)
-vf: Video filter
-af: Audio filter
-t: Duration limit
-ss: Start time
-r: Frame rate
-b:v: Video bitrate
-b:a: Audio bitrate

EXAMPLES:

COMPRESSION:
Input: "compress sample.png"
Output: ffmpeg -i sample.png -c:v libwebp -lossless 0 -q 80 sample_compressed.webp -y

Input: "compress video.mp4"
Output: ffmpeg -i video.mp4 -c:v libx264 -crf 23 -preset slow -c:a libmp3lame -b:a 192k video_compressed.mp4 -y

IMAGE TO IMAGE CONVERSIONS:
Input: "convert sample.png to jpg"
Output: ffmpeg -i sample.png sample.jpg -y

Input: "convert photo.jpg to png"
Output: ffmpeg -i photo.jpg photo.png -y

Input: "convert image.png to webp"
Output: ffmpeg -i image.png -c:v libwebp image.webp -y

Input: "convert file.webp to jpg"
Output: ffmpeg -i file.webp file.jpg -y

Input: "convert picture.jpg to webp"
Output: ffmpeg -i picture.jpg -c:v libwebp picture.webp -y

Input: "convert photo.png to gif"
Output: ffmpeg -i photo.png -vf scale=800:600 photo.gif -y

Input: "convert image.gif to png"
Output: ffmpeg -i image.gif image.png -y

IMAGE TO VIDEO CONVERSIONS:
Input: "convert image.png to video"
Output: ffmpeg -i image.png -vf scale=1920:1080 -c:v libx264 -crf 18 -pix_fmt yuv420p image.mp4 -y

Input: "convert photo.jpg to mp4"
Output: ffmpeg -i photo.jpg -vf scale=1920:1080 -c:v libx264 -crf 18 -pix_fmt yuv420p photo.mp4 -y

VIDEO TO VIDEO CONVERSIONS:
Input: "convert video.mp4 to mov"
Output: ffmpeg -i video.mp4 -c:v copy -c:a copy video.mov -y

Input: "convert movie.mp4 to avi"
Output: ffmpeg -i movie.mp4 -c:v libx264 -c:a libmp3lame movie.avi -y

Input: "convert clip.mov to mp4"
Output: ffmpeg -i clip.mov -c:v copy -c:a copy clip.mp4 -y

Input: "convert video.avi to mp4"
Output: ffmpeg -i video.avi -c:v libx264 -c:a libmp3lame video.mp4 -y

VIDEO TO IMAGE CONVERSIONS:
Input: "convert video.mp4 to png"
Output: ffmpeg -i video.mp4 -vf "select=eq(n\\,0)" -vframes 1 video.png -y

Input: "convert clip.mp4 to jpg"
Output: ffmpeg -i clip.mp4 -vf "select=eq(n\\,0)" -vframes 1 clip.jpg -y

AUDIO CONVERSIONS:
Input: "convert audio.mp3 to wav"
Output: ffmpeg -i audio.mp3 audio.wav -y

Input: "convert sound.wav to mp3"
Output: ffmpeg -i sound.wav -c:a libmp3lame -b:a 192k sound.mp3 -y

CROPPING:
Input: "crop sample.png"
Output: ffmpeg -i sample.png -vf crop=in_w/2:in_h/2:in_w/4:in_h/4 sample_cropped.png -y

Input: "crop video.mp4 to 640x480"
Output: ffmpeg -i video.mp4 -vf crop=640:480:0:0 video_cropped.mp4 -y

CONVERSION RULES (CRITICAL):

IMAGE TO IMAGE:
- "convert X.png to jpg" or "convert X.png to jpeg" → ffmpeg -i X.png X.jpg -y
- "convert X.jpg to png" → ffmpeg -i X.jpg X.png -y
- "convert X.png to webp" → ffmpeg -i X.png -c:v libwebp X.webp -y
- "convert X.jpg to webp" → ffmpeg -i X.jpg -c:v libwebp X.webp -y
- "convert X.webp to jpg" → ffmpeg -i X.webp X.jpg -y
- "convert X.webp to png" → ffmpeg -i X.webp X.png -y
- "convert X.png to gif" → ffmpeg -i X.png -vf scale=800:600 X.gif -y
- "convert X.jpg to gif" → ffmpeg -i X.jpg -vf scale=800:600 X.gif -y
- "convert X.gif to png" → ffmpeg -i X.gif X.png -y
- "convert X.gif to jpg" → ffmpeg -i X.gif X.jpg -y

IMAGE TO VIDEO:
- "convert X to video" or "convert X to mp4" → ffmpeg -i X -vf scale=1920:1080 -c:v libx264 -crf 18 -pix_fmt yuv420p X.mp4 -y
- "convert X.png to mov" → ffmpeg -i X.png -vf scale=1920:1080 -c:v libx264 -crf 18 -pix_fmt yuv420p X.mov -y

VIDEO TO VIDEO:
- "convert X.mp4 to mov" → ffmpeg -i X.mp4 -c:v copy -c:a copy X.mov -y
- "convert X.mp4 to avi" → ffmpeg -i X.mp4 -c:v libx264 -c:a libmp3lame X.avi -y
- "convert X.mov to mp4" → ffmpeg -i X.mov -c:v copy -c:a copy X.mp4 -y
- "convert X.avi to mp4" → ffmpeg -i X.avi -c:v libx264 -c:a libmp3lame X.mp4 -y
- For video→video: use -c copy when possible for speed, or appropriate codec for target format

VIDEO TO IMAGE:
- "convert X.mp4 to png" → ffmpeg -i X.mp4 -vf "select=eq(n\\,0)" -vframes 1 X.png -y
- "convert X.mp4 to jpg" → ffmpeg -i X.mp4 -vf "select=eq(n\\,0)" -vframes 1 X.jpg -y

AUDIO:
- "convert X.mp3 to wav" → ffmpeg -i X.mp3 X.wav -y
- "convert X.wav to mp3" → ffmpeg -i X.wav -c:a libmp3lame -b:a 192k X.mp3 -y

GENERAL RULES:
- NEVER use compression codecs (libwebp, -crf) for format conversions unless converting TO that format
- For PNG→MP4: use -vf scale and -c:v libx264 with -pix_fmt yuv420p, output .mp4 extension
- For PNG→GIF: use -vf scale (no -loop needed for single image), output .gif extension
- For WebP→JPG: simple format conversion, output .jpg extension
- For image→image: usually simple format conversion, use -c:v libwebp only when converting TO webp

CROPPING RULES:
- "crop X" → output center crop: ffmpeg -i X -vf crop=in_w/2:in_h/2:in_w/4:in_h/4 X_cropped.EXT -y
- "crop X to WxH" → output specific crop: ffmpeg -i X -vf crop=W:H:0:0 X_cropped.EXT -y
- Always use -vf crop=WIDTH:HEIGHT:X:Y format
- Output filename should end with _cropped.EXT

COMPRESSION vs CONVERSION vs CROPPING:
- COMPRESSION: "compress X" → use compression codecs (libwebp, libx264 with -crf) to reduce file size
- CONVERSION: "convert X to Y" → change format, use appropriate codec for target format
- CROPPING: "crop X" → use -vf crop filter to extract a portion of the video/image

OUTPUT FORMAT (CRITICAL):
- Output format: ffmpeg -i filename [options] output_filename -y
- Example: ffmpeg -i sample.png -c:v libwebp -lossless 0 -q 80 sample_compressed.webp -y
- DO NOT include any text before or after the command
- DO NOT say "Here's the command:" or similar
- DO NOT use markdown code blocks
- Output the COMPLETE command starting with "ffmpeg"
- Use EXACT filenames from the directory listing provided in the context

REMEMBER: You are a command generator, not a conversational assistant. Output ONLY the complete ffmpeg command.
"""

MAX_CORRECTION_ATTEMPTS = 20  # Increased to allow multiple retries until valid file is generated


class FFmpegExecutor:
    """Orchestrates LLM-based ffmpeg command generation and execution."""

    def __init__(self, *, llm: LLMClient, options: CLIOptions):
        self.llm = llm
        self.options = options

    def run_once(self, user_request: str) -> str:
        """Execute a single ffmpeg command request with LLM-powered validation and error analysis.
        Continues until a valid, non-zero output file is generated."""
        messages = self._build_messages(user_request)
        last_error: Exception | None = None
        last_command: str | None = None
        attempt = 0

        while attempt < MAX_CORRECTION_ATTEMPTS:
            attempt += 1
            try:
                response = self.llm.complete(messages)
                
                # Show thinking content if available (some APIs include reasoning)
                if hasattr(self.llm, "_last_thinking_content") and self.llm._last_thinking_content:
                    thinking = self.llm._last_thinking_content
                    console.print(f"[dim]Thinking:[/dim] {thinking[:500]}..." if len(thinking) > 500 else f"[dim]Thinking:[/dim] {thinking}")
                
                # Show model output preview
                console.print(f"[dim]Model Output:[/dim] {response[:500]}..." if len(response) > 500 else f"[dim]Model Output:[/dim] {response}")
                
                # Check if model wants to ask user for file selection
                response_upper = response.upper()
                if ("ASK_USER_FOR_FILE" in response_upper or 
                    "ask user" in response.lower() or 
                    "need to ask" in response.lower() or
                    "which file" in response.lower() or
                    ("ambiguous" in response.lower() and "file" in response.lower())):
                    console.print("\n[yellow]When info is less, look here is the list of available files:[/yellow]")
                    selected_file = self._interactive_file_selection(
                        self.options.runtime.workdir,
                        "Select one"
                    )
                    if selected_file:
                        # Update the user request with the selected file
                        updated_request = f"{user_request} (using file: {selected_file})"
                        messages = self._build_messages(updated_request)
                        continue
                    else:
                        raise ValueError("File selection was cancelled or failed")
                
                # Extract command from response
                command = self._parse_ffmpeg_command(response)
                last_command = command
                
                console.print(f"[cyan]Attempt {attempt}:[/cyan] {command}")
                
                # Execute command and validate output file
                result = self._execute_command(command)
                
                # Double-check output file exists and is non-zero (extra safety)
                output_file = self._extract_output_filename(command)
                if output_file:
                    # Clean up any remaining quotes or whitespace
                    output_file = output_file.strip().strip('"\'')
                    output_path = self.options.runtime.workdir / output_file
                    
                    # Check if file exists and has content
                    if output_path.exists():
                        file_size = output_path.stat().st_size
                        if file_size > 0:
                            console.print(f"[green]Success![/green] Generated '{output_file}' ({file_size} bytes)")
                            return result
                        else:
                            # File exists but is empty
                            error_msg = f"Output file '{output_file}' exists but is empty (0 bytes)"
                            console.print(f"[red]File Validation Failed:[/red] {error_msg}")
                    else:
                        # File doesn't exist - check if it might be in a different location
                        # List files in workdir to help debug
                        try:
                            existing_files = list(self.options.runtime.workdir.glob("*"))
                            similar_files = [f.name for f in existing_files if output_file.lower() in f.name.lower() or f.name.lower() in output_file.lower()]
                            if similar_files:
                                error_msg = f"Output file '{output_file}' was not found. Similar files found: {', '.join(similar_files[:3])}"
                            else:
                                error_msg = f"Output file '{output_file}' was not created in {self.options.runtime.workdir}"
                        except Exception:
                            error_msg = f"Output file '{output_file}' was not created"
                        
                        console.print(f"[red]File Validation Failed:[/red] {error_msg}")
                    
                    messages.append(
                        {
                            "role": "assistant",
                            "content": command,
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": f"ERROR (Attempt {attempt}): {error_msg}\n\nThe command executed but the output file validation failed. Debug this issue, reason about what went wrong, and provide a corrected ffmpeg command for: {user_request}",
                        }
                    )
                    continue
                
                return result
                
            except (LLMClientError, ValueError) as exc:
                last_error = exc
                error_msg = str(exc)
                console.print(f"[red]Error (Attempt {attempt}):[/red] {error_msg}")
                
                messages.append(
                    {
                        "role": "assistant",
                        "content": last_command if last_command else f"Previous attempt failed: {error_msg[:200]}",
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"ERROR (Attempt {attempt}): {error_msg[:500]}\n\nDebug this error, reason about what went wrong, and provide a corrected ffmpeg command for: {user_request}",
                    }
                )
                
            except subprocess.CalledProcessError as exc:
                last_error = exc
                error_msg = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode() if exc.stderr else str(exc))
                # Extract just the actual error, not version info
                error_lines = error_msg.split("\n")
                actual_errors = [line for line in error_lines if not any(skip in line.lower() for skip in ["copyright", "built with", "configuration:", "libav", "ffmpeg version"])]
                clean_error = "\n".join(actual_errors[-5:])  # Last 5 error lines
                console.print(f"[red]FFmpeg Error (Attempt {attempt}):[/red]\n{clean_error}")
                
                messages.append(
                    {
                        "role": "assistant",
                        "content": last_command if last_command else f"Previous command failed with error: {clean_error[:300]}",
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"ERROR (Attempt {attempt}): FFmpeg Error: {clean_error[:500]}\n\nDebug this ffmpeg error, reason about what went wrong, and provide a corrected command for: {user_request}",
                    }
                )

        # If we've exhausted attempts, raise with last error
        if last_error:
            raise ValueError(f"Failed after {attempt} attempts. Last error: {last_error}")
        raise ValueError(f"Failed after {attempt} attempts without generating a valid output file")

    def repl(self) -> None:
        """Interactive REPL mode."""
        console.print("[bold green]llmpeg ready. Type 'exit' to quit.[/bold green]")
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            except (KeyboardInterrupt, EOFError):
                console.print("\nExiting.")
                break
            if user_input.strip().lower() in {"exit", "quit"}:
                console.print("Goodbye!")
                break
            if not user_input.strip():
                continue

            try:
                result = self.run_once(user_input)
                console.print(f"[bold green]Success:[/bold green] {result}")
            except Exception as exc:
                console.print(f"[red]Error:[/red] {exc}")

    def _build_messages(self, user_request: str) -> List[dict[str, str]]:
        """Build conversation messages with directory context."""
        workdir = self.options.runtime.workdir
        directory_listing = self._get_directory_listing(workdir)

        # Detect request type
        is_conversion = "convert" in user_request.lower() and "to" in user_request.lower()
        is_compression = "compress" in user_request.lower()
        is_cropping = "crop" in user_request.lower()
        
        if is_conversion:
            request_type = "CONVERSION"
        elif is_compression:
            request_type = "COMPRESSION"
        elif is_cropping:
            request_type = "CROPPING"
        else:
            request_type = "UNKNOWN"
        
        # Build a focused prompt with clear structure
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"REQUEST TYPE: {request_type}\n"
            f"CURRENT DIRECTORY: {workdir}\n"
            f"AVAILABLE FILES (numbered for easy reference):\n{directory_listing}\n\n"
            f"USER REQUEST: {user_request}\n\n"
            f"IMPORTANT: If the user request is ambiguous or doesn't clearly specify which file to use, "
            f"output 'ASK_USER_FOR_FILE' instead of guessing. The system will then show the user "
            f"the list of available files to select from.\n"
        )
        
        if is_conversion:
            # Extract target format from request
            target_format = None
            if "to video" in user_request.lower() or "to mp4" in user_request.lower():
                target_format = "mp4"
            elif "to jpg" in user_request.lower() or "to jpeg" in user_request.lower():
                target_format = "jpg"
            elif "to gif" in user_request.lower():
                target_format = "gif"
            elif "to webp" in user_request.lower():
                target_format = "webp"
            
            prompt += (
                f"CONVERSION REQUEST: Convert to {target_format or 'different format'}.\n"
                f"Output filename MUST end with .{target_format if target_format else 'ext'}\n"
                f"For video (PNG→MP4): use -vf scale=1920:1080 -c:v libx264 -crf 18 -pix_fmt yuv420p\n"
                f"For GIF (PNG→GIF): use -vf scale=800:600 (no -loop needed for single image)\n"
                f"For images (format change): use simple format conversion\n"
            )
        elif is_cropping:
            prompt += (
                "CROPPING REQUEST: Crop the video/image.\n"
                "Use -vf crop=WIDTH:HEIGHT:X:Y format.\n"
                "For center crop: use crop=in_w/2:in_h/2:in_w/4:in_h/4\n"
                "Output filename should end with _cropped.EXT\n"
            )
        else:
            prompt += (
                "COMPRESSION REQUEST: Reduce file size.\n"
            )
        
        prompt += "\nOutput ONLY the complete ffmpeg command (no explanations, no text, just the command starting with 'ffmpeg'):"

        return [{"role": "user", "content": prompt}]

    def _get_directory_listing(self, workdir: Path, limit: int = 100) -> str:
        """Get a formatted listing of directory contents with numbered files."""
        try:
            entries = sorted(workdir.iterdir(), key=lambda p: p.name)
            lines = []
            file_count = 0
            for entry in entries[:limit]:
                kind = "dir" if entry.is_dir() else "file"
                size = ""
                if entry.is_file():
                    file_count += 1
                    try:
                        size_bytes = entry.stat().st_size
                        if size_bytes < 1024:
                            size = f" ({size_bytes}B)"
                        elif size_bytes < 1024 * 1024:
                            size = f" ({size_bytes // 1024}KB)"
                        else:
                            size = f" ({size_bytes // (1024 * 1024)}MB)"
                    except Exception:
                        pass
                    lines.append(f"  {file_count}. {entry.name} ({kind}){size}")
                else:
                    lines.append(f"  {entry.name}/ ({kind})")
            if len(entries) > limit:
                lines.append(f"  ... and {len(entries) - limit} more items")
            return "\n".join(lines) if lines else "  (empty directory)"
        except Exception as exc:
            return f"  (error listing directory: {exc})"
    
    def _get_file_list(self, workdir: Path) -> List[Path]:
        """Get a list of files (not directories) in the working directory."""
        try:
            return sorted([p for p in workdir.iterdir() if p.is_file()], key=lambda p: p.name)
        except Exception:
            return []
    
    def _interactive_file_selection(self, workdir: Path, prompt_text: str = "Select a file") -> str | None:
        """Show interactive file selection prompt and return selected filename."""
        files = self._get_file_list(workdir)
        if not files:
            console.print("[yellow]No files found in the current directory.[/yellow]")
            return None
        
        console.print(f"\n[bold cyan]Available files:[/bold cyan]")
        for i, file_path in enumerate(files, 1):
            try:
                size_bytes = file_path.stat().st_size
                if size_bytes < 1024:
                    size = f"{size_bytes}B"
                elif size_bytes < 1024 * 1024:
                    size = f"{size_bytes // 1024}KB"
                else:
                    size = f"{size_bytes // (1024 * 1024)}MB"
            except Exception:
                size = "unknown"
            console.print(f"  [cyan]{i}.[/cyan] {file_path.name} ({size})")
        
        try:
            choice = Prompt.ask(f"\n[bold]{prompt_text}[/bold]", default="1")
            choice_num = int(choice.strip())
            if 1 <= choice_num <= len(files):
                selected_file = files[choice_num - 1]
                console.print(f"[green]Selected:[/green] {selected_file.name}")
                return selected_file.name
            else:
                console.print(f"[red]Invalid selection. Please choose 1-{len(files)}[/red]")
                return None
        except (ValueError, KeyboardInterrupt):
            console.print("[yellow]Selection cancelled.[/yellow]")
            return None

    def _parse_ffmpeg_command(self, text: str) -> str:
        """Parse and clean ffmpeg command from text."""

        # Remove conversational prefixes
        text = re.sub(r"^(I'm sorry|Sorry|I can't|I cannot|Here's|Here is|The command|Command|Output|Result):\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove error message prefixes
        text = re.sub(r"^(Previous attempt failed|Error|Failed):\s*", "", text, flags=re.IGNORECASE)
        
        # Extract content from markdown code blocks
        code_block_match = re.search(r"```(?:bash|sh|ffmpeg)?\s*\n(.*?)```", text, re.DOTALL)
        if code_block_match:
            text = code_block_match.group(1).strip()
        
        # Remove markdown code block markers if still present
        text = re.sub(r"```(?:bash|sh|ffmpeg)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        
        # Remove conversational suffixes (e.g., "Please check", "Let me know", etc.)
        text = re.sub(r"\s+(Please|Let me|I'll|I will|You can|Try|Note|Remember).*$", "", text, flags=re.IGNORECASE | re.DOTALL)

        # Keep "ffmpeg" prefix if present (we want the full command now)
        # But remove it temporarily for extraction, then add it back
        has_ffmpeg_prefix = text.strip().lower().startswith("ffmpeg")
        if has_ffmpeg_prefix:
            text = re.sub(r"^ffmpeg\s+", "", text, flags=re.IGNORECASE)
        
        # Remove "convert" word if model mistakenly includes it (it's not an ffmpeg flag)
        text = re.sub(r"^convert\s+", "", text, flags=re.IGNORECASE)
        
        # Remove "to X" phrases that might be included (e.g., "to video", "to jpg")
        text = re.sub(r"\s+to\s+(video|mp4|jpg|jpeg|gif|webp|png)\s*", " ", text, flags=re.IGNORECASE)
        
        # Remove Python imports and other non-command lines
        lines = text.split("\n")
        filtered_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip Python imports
            if line.startswith("import ") or line.startswith("from "):
                continue
            # Skip shebang
            if line.startswith("#!"):
                continue
            # Skip comments
            if line.startswith("#"):
                continue
            filtered_lines.append(line)
        text = "\n".join(filtered_lines)

        # Split into lines and find the most likely command line
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            raise ValueError("No command found in LLM response")

        # Look for a line that starts with ffmpeg or -i (input) or contains common ffmpeg flags
        command = None
        for line in lines:
            # Skip shebang lines
            if line.startswith("#!"):
                continue
            # Skip comment lines
            if line.startswith("#"):
                continue
            # Skip error message lines and conversational text
            if any(skip in line.lower() for skip in [
                "previous attempt", "error:", "failed:", "command missing",
                "i'm sorry", "sorry", "i can't", "i cannot", "here's", "here is",
                "please", "let me", "you can", "try", "note", "remember",
                "the command", "attempted to", "provided filename", "does not match",
                "<think>", "redacted_reasoning", "thinking content"
            ]):
                continue
            # Skip version/build info lines
            if any(skip in line.lower() for skip in ["copyright", "built with", "configuration:", "libav", "ffmpeg version"]):
                continue
            
            # Extract ffmpeg command from line (might be embedded in text)
            # Look for "ffmpeg" followed by command-like patterns
            ffmpeg_match = re.search(r"(ffmpeg\s+[^\n]+)", line, re.IGNORECASE)
            if ffmpeg_match:
                command = ffmpeg_match.group(1).strip()
                # Clean up: remove trailing punctuation/explanations
                command = re.sub(r"[.,;:!?].*$", "", command)
                break
            
            # Remove "convert" word if present (not an ffmpeg flag)
            line_clean = re.sub(r"^convert\s+", "", line, flags=re.IGNORECASE)
            # Look for lines with ffmpeg-like patterns (must start with ffmpeg, -i, or contain common flags)
            if re.search(r"(^ffmpeg\s+|^-i\s+\S+|-[cv]|-[ab]|scale=|crf=|-q|-preset)", line_clean):
                # Make sure it's not just version info
                if not re.search(r"(version|copyright|built|configuration)", line_clean, re.IGNORECASE):
                    command = line_clean
                    break

        # If no pattern match, take the first non-comment line that looks like a command
        if not command:
            for line in lines:
                if (not line.startswith("#") and 
                    not line.startswith("!") and
                    not any(skip in line.lower() for skip in ["copyright", "built", "configuration", "libav", "version"]) and
                    ("ffmpeg" in line.lower() or "-" in line or ".mp4" in line or ".png" in line or ".jpg" in line or ".webp" in line)):
                    command = line
                    break

        if not command:
            raise ValueError(f"No valid command found in response: {text[:200]}")

        # Remove quotes if wrapped
        if command.startswith('"') and command.endswith('"'):
            command = command[1:-1]
        if command.startswith("'") and command.endswith("'"):
            command = command[1:-1]

        # Ensure command starts with "ffmpeg"
        if not command.lower().startswith("ffmpeg"):
            # If it starts with -i, prepend ffmpeg
            if command.strip().startswith("-i"):
                command = f"ffmpeg {command}"
            else:
                # Try to add ffmpeg prefix
                command = f"ffmpeg {command}"

        # Validate it looks like an ffmpeg command
        if not command or len(command) < 10:
            raise ValueError(f"Command too short or invalid: {command}")

        # Check for placeholder tokens
        placeholder_patterns = ["<", ">", "{{", "}}"]
        if any(pattern in command for pattern in placeholder_patterns):
            raise ValueError(f"Command contains placeholders: {command}")

        # Light auto-fix: ensure flags have proper spacing
        command = re.sub(r"-i([^/\s-])", r"-i \1", command)  # Fix "-i./file" -> "-i ./file"
        command = re.sub(r"-y([^/\s-])", r"-y \1", command)  # Fix "-y./file" -> "-y ./file"
        command = re.sub(r"-o([^/\s-])", r"\1", command)  # Remove -o flag (ffmpeg doesn't use -o)

        # Validate command has output filename (must end with a filename, not just flags)
        # Check if command ends with a file extension or -y flag followed by filename
        parts = command.split()
        has_output_file = False
        for i, part in enumerate(parts):
            # Check if this part looks like an output filename (has extension and not a flag)
            if (not part.startswith("-") and 
                any(ext in part.lower() for ext in [".mp4", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp3", ".mov", ".avi"])):
                # Make sure it's not the input file (input file comes right after -i)
                if i > 0 and parts[i-1] != "-i":
                    has_output_file = True
                    break
        
        if not has_output_file:
            raise ValueError(f"Command missing output filename: {command}")
        
        # Check for version/build info contamination
        if any(bad in command.lower() for bad in ["copyright", "built with", "configuration:", "libavutil", "libavcodec"]):
            raise ValueError(f"Command contains version info, not a real command: {command[:100]}")

        return command.strip()
    
    def _validate_conversion_match(self, command: str, user_request: str) -> None:
        """Validate that conversion commands match the requested output format."""
        user_lower = user_request.lower()
        
        # Check for conversion requests
        if "convert" in user_lower and "to" in user_lower:
            # Extract target format
            if "to video" in user_lower or "to mp4" in user_lower:
                if ".mp4" not in command.lower():
                    raise ValueError(f"Conversion to video requested but command doesn't output .mp4: {command}")
            elif "to jpg" in user_lower or "to jpeg" in user_lower:
                if ".jpg" not in command.lower() and ".jpeg" not in command.lower():
                    raise ValueError(f"Conversion to JPG requested but command doesn't output .jpg: {command}")
            elif "to gif" in user_lower:
                if ".gif" not in command.lower():
                    raise ValueError(f"Conversion to GIF requested but command doesn't output .gif: {command}")
            elif "to webp" in user_lower:
                if ".webp" not in command.lower():
                    raise ValueError(f"Conversion to WebP requested but command doesn't output .webp: {command}")

    def _execute_command(self, command: str) -> str:
        """Execute ffmpeg command and verify success."""
        if self.options.runtime.dry_run:
            console.print(f"[yellow]DRY-RUN:[/yellow] {command}")
            return "Command would be executed (dry-run mode)"

        # Extract output filename from command to validate it's created
        output_file = self._extract_output_filename(command)
        
        # Command already includes "ffmpeg" prefix, so split it properly
        # Remove "ffmpeg" prefix and split the rest
        if command.lower().startswith("ffmpeg"):
            command_args = command[6:].strip()  # Remove "ffmpeg" prefix
            argv = ["ffmpeg"] + shlex.split(command_args)
        else:
            # Fallback: assume it's just arguments
            argv = ["ffmpeg"] + shlex.split(command)

        if self.options.runtime.verbose:
            console.print(f"[dim]Executing:[/dim] {' '.join(argv)}")

        # Execute using subprocess.run
        process = subprocess.run(
            argv,
            cwd=self.options.runtime.workdir,
            capture_output=True,
            text=True,
            check=True,
        )

        stdout = process.stdout.strip()
        stderr = process.stderr.strip()

        # Verify success
        if process.returncode != 0:
            error_msg = stderr or stdout or "Unknown error"
            raise subprocess.CalledProcessError(
                process.returncode, argv, stdout, stderr
            )

        # Validate output file was created and has non-zero size
        if output_file:
            output_path = self.options.runtime.workdir / output_file
            if not output_path.exists():
                raise ValueError(f"Output file '{output_file}' was not created")
            if output_path.stat().st_size == 0:
                raise ValueError(f"Output file '{output_file}' is empty (0 bytes). The command may have failed silently.")

        return stdout or "Command completed successfully"
    
    def _extract_output_filename(self, command: str) -> str | None:
        """Extract output filename from ffmpeg command."""
        try:
            # Use shlex.split to properly handle quoted filenames with spaces
            parts = shlex.split(command)
        except ValueError:
            # Fallback to simple split if shlex fails (shouldn't happen with valid commands)
            parts = command.split()
        
        output_file = None
        
        # Find position of -y flag (output file is usually right before -y)
        y_index = -1
        for i, part in enumerate(parts):
            if part == "-y":
                y_index = i
                break
        
        # Look backwards from -y for the output filename
        if y_index > 0:
            for i in range(y_index - 1, -1, -1):
                part = parts[i]
                # Skip flags and their values
                if part.startswith("-"):
                    continue
                # Check if it's a filename (has extension)
                if any(ext in part.lower() for ext in [".mp4", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp3", ".mov", ".avi", ".mkv", ".wav"]):
                    # Remove all quotes (handles cases like 'file.mp4" or "file.mp4')
                    output_file = part.strip().strip('"\'')
                    break
        
        # Fallback: if no -y flag, find the last filename argument
        if not output_file:
            # Track input file position
            input_index = -1
            for i, part in enumerate(parts):
                if part == "-i" and i + 1 < len(parts):
                    input_index = i + 1
                    break
            
            # Find the last filename (after input file)
            for i in range(len(parts) - 1, -1, -1):
                if i <= input_index:
                    break
                part = parts[i]
                if not part.startswith("-") and any(ext in part.lower() for ext in [".mp4", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp3", ".mov", ".avi", ".mkv", ".wav"]):
                    # Remove all quotes (handles cases like 'file.mp4" or "file.mp4')
                    output_file = part.strip().strip('"\'')
                    break
        
        return output_file


__all__ = ["FFmpegExecutor"]
