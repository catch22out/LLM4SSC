
class Prompt():
    def __init__(self):
        self.instruction = ""
        self.patch = ""
        self.adapt_code = ""

        self.prompt = ""
        
    def prompt_generate(self):
        self.instruction = "You're a professional and cautious C programmer, and you're very good at patching programs. Now I'm going to give you a patch and a piece of code to fix, but it's worth noting that the patch you've been given won't necessarily work directly with this code; you'll need to adapt it. Your modification can only be made between two comment lines. You only need to give the repaired code between the two comment lines. Do not output anyother things!"

        self.patch = """
        Patch:
        int hw, ap, ap_max = ie[1];
        u8 hw_rate;
        + if (ap_max > MAX_RATES) {
        +     lbs_deb_assoc("invalid rates\n");
        +     return tlv;
        + }
        ie += 2;
        lbs_deb_hex(LBS_DEB_ASSOC, "AP IE Rates", (u8 *) ie, ap_max);
        """

        self.adapt_code = """
        Code to be fixed:
        int hw, ap, ap_max = ie[1];
        u8 hw_rate;
        /* The area below that needs to be repaired */
        
        /* The upper area that needs to be repaired */
        ie += 2;
        lbs_deb_hex(LBS_DEB_ASSOC, "AP IE Rates", (u8 *) ie, ap_max);
        """
    
        self.prompt = "\n\n".join([self.instruction, self.patch, self.adapt_code])

        return self.prompt