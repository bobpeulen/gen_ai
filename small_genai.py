compartment_id = "ocid1.compartment.oc1..aaaaaaaae3n6r6hrjipbap2hojicrsvkzatrtlwvsyrpyjd7wjnw4za3m75q"
CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)


def predict_genai(input_text):

    # Service endpoint
    endpoint = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"

    generative_ai_client = oci.generative_ai.GenerativeAiClient(config=config, service_endpoint=endpoint, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))

    prompts = [input_text]
    generate_text_detail = oci.generative_ai.models.GenerateTextDetails()
    generate_text_detail.prompts = prompts
    generate_text_detail.serving_mode = oci.generative_ai.models.OnDemandServingMode(model_id="cohere.command")
    # generate_text_detail.serving_mode = oci.generative_ai.models.DedicatedServingMode(endpoint_id="custom-model-endpoint") # for custom model from Dedicated AI Cluster
    generate_text_detail.compartment_id = compartment_id
    generate_text_detail.max_tokens = 20
    generate_text_detail.temperature = 0.75
    generate_text_detail.frequency_penalty = 1.0
    generate_text_detail.top_p = 0.7

    generate_text_response = generative_ai_client.generate_text(generate_text_detail)

    # Print result
    print("**************************Generate Texts Result**************************")
    print(generate_text_response.data)
    print("Response from GenAI = " + generate_text_response.data.generated_texts[0][0].text)
    output_text = generate_text_response.data.generated_texts[0][0].text

    return output_text
