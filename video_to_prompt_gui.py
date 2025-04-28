import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
import ffmpeg # تأكد من تثبيت: pip install ffmpeg-python
import torch # تأكد من تثبيت torch و torchvision
import threading
import queue
import time

# --- تحميل BLIP مبدئياً (يتم عند أول استخدام لتسريع بدء التشغيل) ---
# تأكد من تثبيت: pip install transformers Pillow
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("تحذير: مكتبة transformers غير مثبتة. لن تعمل ميزة توليد البرومبت.")

blip_processor = None
blip_model = None
blip_device = None
blip_model_name = "Salesforce/blip-image-captioning-base"

def load_blip_model():
    """Loads the BLIP model and processor if not already loaded."""
    global blip_processor, blip_model, blip_device
    if blip_model is None or blip_processor is None:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("مكتبة transformers غير متاحة.")
        try:
            print("جاري تحميل موديل BLIP (قد يستغرق بعض الوقت)...")
            blip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"استخدام الجهاز لـ BLIP: {blip_device}")
            blip_processor = BlipProcessor.from_pretrained(blip_model_name)
            blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name)
            blip_model.to(blip_device)
            blip_model.eval() # مهم لوضع الموديل في وضع التقييم
            print("تم تحميل موديل BLIP بنجاح.")
        except Exception as e:
            print(f"خطأ أثناء تحميل موديل BLIP: {e}")
            blip_model = None # منع المحاولات المستقبلية إذا فشل
            blip_processor = None
            raise e # إعادة رفع الخطأ ليتم التعامل معه
    return blip_processor, blip_model, blip_device

class VideoToPromptApp:
    def __init__(self, master_root):
        self.root = master_root
        self.root.title("Video to Prompt")
        self.root.geometry("450x300") # حجم أكبر قليلاً

        self.video_path = ""
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.update_queue = queue.Queue()

        # استخدام ttk لمظهر أفضل
        self.style = ttk.Style()
        try:
            # حاول استخدام ثيم يتناسب مع النظام
            self.style.theme_use('vista' if os.name == 'nt' else 'clam')
        except tk.TclError:
            print("لم يتم العثور على ثيم الواجهة المفضل، استخدام الافتراضي.")


        # --- إطار للمحتوى ---
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        # --- اختيار الفيديو ---
        self.select_frame = ttk.Frame(self.main_frame)
        self.select_frame.pack(fill=tk.X, pady=5)
        self.select_button = ttk.Button(self.select_frame, text="📁 اختيار فيديو", command=self.select_video)
        self.select_button.pack(side=tk.LEFT, padx=(0, 5))
        self.video_label = ttk.Label(self.select_frame, text="لم يتم اختيار فيديو بعد.", width=40, relief=tk.SUNKEN)
        self.video_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- إعدادات (مثال FPS) ---
        self.settings_frame = ttk.Frame(self.main_frame)
        self.settings_frame.pack(fill=tk.X, pady=5)
        self.fps_label = ttk.Label(self.settings_frame, text="استخراج إطار كل (ثانية):")
        self.fps_label.pack(side=tk.LEFT, padx=(0, 5))
        self.fps_var = tk.IntVar(value=1) # القيمة الافتراضية 1 إطار/ثانية
        self.fps_spinbox = ttk.Spinbox(self.settings_frame, from_=1, to=60, textvariable=self.fps_var, width=5)
        self.fps_spinbox.pack(side=tk.LEFT)
        # (يمكن إضافة المزيد من الإعدادات هنا، مثل اسم مجلد الإخراج)

        # --- أزرار التحكم ---
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=10)
        self.run_button = ttk.Button(self.control_frame, text="🚀 تحويل إلى برومبت", command=self.start_processing, state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(self.control_frame, text="🛑 إيقاف", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # --- التقدم والحالة ---
        self.progress_label = ttk.Label(self.main_frame, text="الحالة: جاهز", anchor=tk.W)
        self.progress_label.pack(fill=tk.X, pady=5)
        self.progress_bar = ttk.Progressbar(self.main_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=5)

        # التحقق الأولي من المكتبات
        if not TRANSFORMERS_AVAILABLE:
             self.progress_label.config(text="تحذير: مكتبة transformers مفقودة!")
             messagebox.showwarning("تحذير", "مكتبة transformers غير مثبتة.\nلن تعمل ميزة توليد البرومبت.\nقم بتثبيتها: pip install transformers torch torchvision Pillow")


    def select_video(self):
        path = filedialog.askopenfilename(title="اختر ملف فيديو", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if path:
            self.video_path = path
            # عرض اسم الملف فقط لتجنب المسارات الطويلة
            self.video_label.config(text=os.path.basename(path))
            self.run_button.config(state=tk.NORMAL if TRANSFORMERS_AVAILABLE else tk.DISABLED) # تفعيل الزر فقط إذا كانت المكتبات متاحة
        else:
            self.video_path = ""
            self.video_label.config(text="لم يتم اختيار فيديو بعد.")
            self.run_button.config(state=tk.DISABLED)

    def start_processing(self):
        if not self.video_path:
            messagebox.showerror("خطأ", "الرجاء اختيار ملف فيديو أولاً.")
            return
        if not TRANSFORMERS_AVAILABLE:
             messagebox.showerror("خطأ", "مكتبة transformers غير متاحة. يرجى تثبيتها.")
             return

        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("قيد التشغيل", "عملية أخرى قيد التشغيل بالفعل.")
            return

        # تحميل الموديل إذا لم يكن محملاً
        try:
            load_blip_model()
        except Exception as e:
            messagebox.showerror("خطأ تحميل الموديل", f"فشل تحميل موديل BLIP:\n{e}")
            return

        # إعادة تعيين حالة الإيقاف والواجهة
        self.stop_event.clear()
        self.run_button.config(state=tk.DISABLED)
        self.select_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_bar['value'] = 0
        self.progress_label.config(text="الحالة: بدء المعالجة...")

        # بدء الـ Thread
        self.processing_thread = threading.Thread(
            target=self._run_processing,
            args=(self.video_path, self.fps_var.get()),
            daemon=True # يموت الـ Thread إذا أُغلق البرنامج الرئيسي
        )
        self.processing_thread.start()

        # بدء التحقق من الـ Queue لتحديث الواجهة
        self.root.after(100, self.check_queue)

    def stop_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_event.set() # إرسال إشارة الإيقاف
            self.stop_button.config(state=tk.DISABLED) # منع الضغط مرة أخرى
            self.progress_label.config(text="الحالة: جاري الإيقاف...")
        else:
            print("لا توجد عملية لإيقافها.")

    def check_queue(self):
        """تفحص الـ Queue وتحديث الواجهة."""
        try:
            while True:
                message = self.update_queue.get_nowait()
                msg_type = message[0]
                msg_data = message[1]

                if msg_type == "status":
                    self.progress_label.config(text=f"الحالة: {msg_data}")
                elif msg_type == "progress":
                    self.progress_bar['value'] = msg_data
                elif msg_type == "result":
                    output_file_path = msg_data
                    messagebox.showinfo("تم بنجاح", f"تم إنشاء البرومبتات بنجاح!\nتم الحفظ في:\n{output_file_path}")
                elif msg_type == "error":
                    messagebox.showerror("حدث خطأ", msg_data)
                elif msg_type == "finished":
                    # إعادة تفعيل الأزرار
                    self.run_button.config(state=tk.NORMAL if self.video_path and TRANSFORMERS_AVAILABLE else tk.DISABLED)
                    self.select_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    self.progress_bar['value'] = 0 # إعادة تصفير شريط التقدم
                    if "إيقاف" not in self.progress_label.cget("text"): # لا تغير الحالة إذا تم الإيقاف
                          self.progress_label.config(text="الحالة: جاهز")
                    return # التوقف عن التحقق الدوري

        except queue.Empty:
            # استمر في التحقق إذا كان الـ Thread لا يزال يعمل
            if self.processing_thread and self.processing_thread.is_alive():
                self.root.after(100, self.check_queue)
            # إذا انتهى الـ Thread دون رسالة finished (قد يحدث عند الإيقاف)
            elif not self.stop_event.is_set(): # تأكد أنه لم يتم إيقافه يدوياً
                 self.run_button.config(state=tk.NORMAL if self.video_path and TRANSFORMERS_AVAILABLE else tk.DISABLED)
                 self.select_button.config(state=tk.NORMAL)
                 self.stop_button.config(state=tk.DISABLED)
                 self.progress_label.config(text="الحالة: انتهى (غير مؤكد)")


    def _run_processing(self, video_path, fps):
        """الوظيفة التي تعمل في الـ Thread المنفصل."""
        output_folder_base = "output_prompts" # مجلد رئيسي للإخراج
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_output_folder = os.path.join(output_folder_base, f"run_{timestamp}")
        frames_folder = os.path.join(run_output_folder, "frames")
        output_prompt_file = os.path.join(run_output_folder, f"video_prompts_{timestamp}.txt")

        try:
            self.update_queue.put(("status", f"إنشاء مجلد الإخراج: {run_output_folder}"))
            os.makedirs(frames_folder, exist_ok=True)

            # --- 1. استخراج الإطارات ---
            self.update_queue.put(("status", "جاري استخراج الإطارات..."))
            self.update_queue.put(("progress", 5)) # نسبة تقدم مبدئية

            process = (
                ffmpeg
                .input(video_path)
                .output(os.path.join(frames_folder, 'frame_%06d.png'), vf=f'fps={fps}')
                .global_args('-progress', 'pipe:1') # لإتاحة قراءة التقدم (قد لا يعمل دائماً)
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
            # يمكنك محاولة قراءة stderr للتقدم هنا، لكنها معقدة، سنعتمد على عدد الملفات لاحقاً
            process.wait() # انتظر انتهاء ffmpeg

            if process.returncode != 0:
                 stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
                 raise ffmpeg.Error('ffmpeg', stdout=None, stderr=stderr_output)


            self.update_queue.put(("status", "اكتمل استخراج الإطارات."))
            self.update_queue.put(("progress", 20))

            # --- 2. توليد البرومبتات ---
            self.update_queue.put(("status", "جاري تحميل الموديل (إذا لزم الأمر)..."))
            # التأكد من تحميل الموديل مرة أخرى داخل الـ Thread (احتياطي)
            processor, model, device = load_blip_model()

            frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".png")])
            num_frames = len(frame_files)
            if num_frames == 0:
                raise ValueError("لم يتم استخراج أي إطارات، تحقق من الفيديو أو إعدادات FPS.")

            self.update_queue.put(("status", f"جاري توليد البرومبتات لـ {num_frames} إطار..."))

            prompts = []
            for i, img_name in enumerate(frame_files):
                # تحقق من طلب الإيقاف
                if self.stop_event.is_set():
                    self.update_queue.put(("status", "تم إيقاف العملية."))
                    # (اختياري: حذف المجلدات المؤقتة عند الإيقاف؟)
                    # import shutil
                    # shutil.rmtree(run_output_folder)
                    self.update_queue.put(("finished", None))
                    return

                img_path = os.path.join(frames_folder, img_name)
                try:
                    image = Image.open(img_path).convert("RGB")

                    # توليد البرومبت
                    # نقل المدخلات للجهاز الصحيح
                    inputs = processor(image, return_tensors="pt").to(device)
                    with torch.no_grad(): # مهم لتعطيل حساب التدرجات
                         out = model.generate(**inputs)
                    caption = processor.decode(out[0], skip_special_tokens=True)

                    # إضافة البرومبت للائحة (يمكن تغيير التنسيق هنا)
                    # frame_number = int(img_name.split('_')[-1].split('.')[0]) # استخراج رقم الإطار
                    prompts.append(f'"{i}": "{caption.strip()}",') # تنسيق مناسب لـ ComfyUI AnimateDiff

                    # تحديث التقدم
                    progress_percent = 20 + int(((i + 1) / num_frames) * 75) # من 20% إلى 95%
                    self.update_queue.put(("progress", progress_percent))
                    self.update_queue.put(("status", f"معالجة الإطار {i+1}/{num_frames}..."))

                except Exception as frame_err:
                    print(f"خطأ في معالجة الإطار {img_name}: {frame_err}")
                    # يمكنك إضافة البرومبت كـ "خطأ" أو تخطيه
                    prompts.append(f'"{i}": "Error processing frame {img_name}",')


            # --- 3. حفظ البرومبتات ---
            self.update_queue.put(("status", "جاري حفظ ملف البرومبتات..."))
            # إزالة الفاصلة الأخيرة
            if prompts:
                prompts[-1] = prompts[-1].rstrip(',')

            with open(output_prompt_file, "w", encoding="utf-8") as f:
                f.write("{\n")
                f.write("\n".join(prompts))
                f.write("\n}")

            self.update_queue.put(("progress", 100))
            self.update_queue.put(("result", output_prompt_file)) # إرسال مسار الملف الناتج

            # --- 4. (اختياري) تنظيف الإطارات ---
            # يمكنك إضافة Checkbox في الواجهة للتحكم بهذا
            # import shutil
            # try:
            #     self.update_queue.put(("status", "جاري حذف الإطارات المؤقتة..."))
            #     shutil.rmtree(frames_folder)
            # except Exception as clean_err:
            #     print(f"تحذير: فشل حذف مجلد الإطارات: {clean_err}")


        except ffmpeg.Error as e:
            print(f"FFmpeg stderr: {e.stderr.decode('utf-8', errors='ignore') if e.stderr else 'No stderr'}")
            self.update_queue.put(("error", f"خطأ FFmpeg: {e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)}"))
        except Exception as e:
            print(f"حدث خطأ غير متوقع: {e}")
            import traceback
            traceback.print_exc() # طباعة تفاصيل الخطأ الكاملة في الطرفية
            self.update_queue.put(("error", f"حدث خطأ: {e}"))
        finally:
            # إرسال إشارة الانتهاء دائماً لتحديث الواجهة
            self.update_queue.put(("finished", None))


if __name__ == "__main__":
    if not TRANSFORMERS_AVAILABLE:
        print("*"*50)
        print(" تحذير: مكتبة transformers غير مثبتة ")
        print(" لن تعمل ميزة توليد البرومبت. ")
        print(" يرجى التثبيت: pip install transformers torch torchvision Pillow ")
        print("*"*50)
        # يمكنك اختيار إغلاق البرنامج هنا إذا أردت
        # sys.exit()

    root = tk.Tk()
    app = VideoToPromptApp(root)
    root.mainloop()