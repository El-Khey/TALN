import tensorflow as tf
from text_processing import TextProcessor

def load_model_and_run_chat(exported_model_dir, use_gpu=False):
    if not use_gpu:
        tf.config.set_visible_devices([], 'GPU')

    text_processor = TextProcessor(None, None)

    print("Chargement du modèle exporté...")
    model = tf.saved_model.load(exported_model_dir)
    translate_fn = model.signatures["serving_default"]

    print("Mini-chat de traduction (entrez 'exit' pour quitter)")
    while True:
        user_input = input("Vous: ")
        if user_input.lower() == "exit":
            print("Fin du chat.")
            break

        # === Prétraitement ===
        tokens = text_processor.tokenize(user_input)
        token_tensor = tf.constant([tokens])       
        length_tensor = tf.constant([len(tokens)]) 

        # === Traduction ===
        result = translate_fn(tokens=token_tensor, length=length_tensor)
        output_tokens = result["tokens"].numpy()[0][0] 
        translation = b" ".join(output_tokens).decode("utf-8")

        print(f"Modèle: {translation}")

if __name__ == "__main__":
    exported_model_dir = "exported_model/"
    use_gpu = False
    load_model_and_run_chat(exported_model_dir, use_gpu)
