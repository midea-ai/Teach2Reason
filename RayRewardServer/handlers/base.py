class BaseHandler:

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def process_batch(self, *args, **kwargs):
        raise NotImplementedError

    def health(self):
        return {
            "ok": True,
            "error": None
        }
