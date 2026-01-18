from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
from .utils import make_gradcam_heatmap
import importlib

class AudioXAIService:

    def __init__(self):
        self.lime_explainer = lime_image.LimeImageExplainer(random_state=42)

    def explain(self, xai, model, grad_model, metadata, image):
        if xai == "LIME":
            try:
                res_base64 = self._explain_with_lime(model, image)
                return {"success": True, "image": res_base64}
            except Exception as e:
                return {"success": False, "error": f"{e}"}
        elif xai == "SHAP":
            try:
                res_base64 = self._explain_with_shap(model, metadata["class_names"], image)
                return {"success": True, "image": res_base64}
            except Exception as e:
                return {"success": False, "error": f"{e}"}
        elif xai == "Grad-CAM":
            try:
                res_base64 = self._explain_with_gradcam(grad_model, metadata["preprocessing"], image)
                return {"success": True, "image": res_base64}
            except Exception as e:
                return {"success": False, "error": f"{e}"}

    def _explain_with_lime(self, model, image):
        image_array = np.array(image)

        explanation = self.lime_explainer.explain_instance(
            image_array, 
            classifier_fn=lambda x: model.predict(x, verbose=0), 
            top_labels=5, 
            hide_color=0, 
            num_samples=1000
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=10, 
            hide_rest=True
        )

        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        ax.imshow(mark_boundaries(temp / 255.0, mask))

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        
        return img_b64


    def _explain_with_shap(self, model, class_names, image):
        image_array = np.array(image)
        X = np.expand_dims(image_array, axis=0)

        masker = shap.maskers.Image("inpaint_telea", image_array.shape)

        explainer = shap.Explainer(model, masker, output_names=class_names)

        shap_values = explainer(X, max_evals=2000, batch_size=50, outputs=shap.Explanation.argsort.flip[:1])

        
        shap.image_plot(shap_values, show=False)
        fig = plt.gcf()
        axes = fig.get_axes()

        fig_new, ax_new = plt.subplots(figsize=(6, 6))

        if len(axes) > 1:
            for img in axes[1].get_images():
                ax_new.imshow(img.get_array(), cmap=img.get_cmap(), 
                            vmin=img.get_clim()[0], vmax=img.get_clim()[1])
            ax_new.axis('off')
        plt.close(fig)

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        
        return img_b64
    
    def _explain_with_gradcam(self, grad_model, preprocessing_import, image,):
        try:
            module = importlib.import_module(preprocessing_import)
            preprocess_func = getattr(module, "preprocess_input")
            print(f"module: {preprocessing_import} loaded successfuly", flush=True)
        except:
            raise ValueError(f"can't load module: {preprocessing_import}")

        image_array = np.array(image)
        X = np.expand_dims(image_array, axis=0)
        X_preprocess = preprocess_func(X.copy())
        
        heatmap = make_gradcam_heatmap(X_preprocess, grad_model, pred_index=None)
        plt.matshow(heatmap)
        plt.axis('off')

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return img_b64


audio_xai_service = AudioXAIService()